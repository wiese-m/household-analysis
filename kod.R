# ---------------------- Biblioteki ----------------------

library(tidyverse)
library(randomForest)
library(mlr)
library(xgboost)
library(caret)
library(Boruta)
library(mlbench)
library(praznik)
library(e1071)
library(Hmisc)
library(class)

# ---------------------- Wczytanie danych ----------------------

X_train <- read.table("data/train_data.txt")
y_train <- read.table("data/train_labels.txt")
individual <- read.table("data/individual_data.txt")
X_test <- read.table("data/test_data.txt")
X_val <- read.table("data/validation_data.txt")
y_val <- read.table("data/validation_labels.txt")

# ---------------------- Analiza i agregacja danych indywidualnych ----------------------

summary(individual)

ind_by_hid <- individual %>%
  group_by(hid) %>%
  summarise(family_size = n(),
            readwrite_pct = sum(ind_readwrite == "Yes") / family_size,
            workers = sum(ind_work4 == "Yes"),
            n_male = sum(ind_sex == "Male"),
            n_female = sum(ind_sex == "Female"),
            age_mean = mean(ind_age),
            school_pct = sum(ind_educ01 == "Yes", na.rm = TRUE) / family_size,
            family_weekly_wh = sum(ind_work1, na.rm = TRUE),
            under10 = sum(ind_age < 10))

summary(ind_by_hid)

# Laczymy dane indywidualne z danymi na poziomie gospodarstwa
X_train <- X_train %>% inner_join(ind_by_hid, by = "hid")
X_val <- X_val %>% inner_join(ind_by_hid, by = "hid")
X_test <- X_test %>% inner_join(ind_by_hid, by = "hid")
train <- data.frame(y_train, X_train)
val <- data.frame(y_val, X_val)

y_train_ind <- train %>%
  arrange(hid) %>%
  select(poor)
ind_train <- ind_by_hid[ind_by_hid$hid %in% X_train$hid, ]
ind_train <- data.frame(y_train_ind, ind_train)

# Sprawdzamy istotnosc nowych zmiennych (wystepuja samodzielnie)
boruta <- Boruta(poor ~ ., data = ind_train[-2], maxRuns = 30, doTrace = 2)
plot(boruta) # wniosek: dodamy te zmienne do modelu

# ---------------------- Wizualizacja danych wielowymiarowych - PCA ----------------------

X_combined <- rbind(X_train, X_val, X_test) # laczymy dane, aby zgadzaly sie poziomy czynnikow
dmy <- dummyVars(" ~ .", data = X_combined[-1])
X_combined_numeric <- data.frame(predict(dmy, newdata = X_combined[-1]))
y_combined <- as.factor(ifelse(rbind(y_train, y_val) == "Non-poor", "Non-poor", "Poor"))

pca <- prcomp(X_combined_numeric[1:9000, ], scale = TRUE)
pca_var <- pca$sdev^2
pca_var_pct <- round(pca_var / sum(pca_var) * 100, 2)
pca_data <- data.frame(pc1 = pca$x[, 1], pc2 = pca$x[, 2], Status = y_combined)

pca_data %>%
  ggplot(aes(pc1, pc2, color = Status)) +
  geom_point() +
  xlab(paste("PC1 - ", pca_var_pct[1], "%", sep = "")) +
  ylab(paste("PC2 - ", pca_var_pct[2], "%", sep = "")) +
  theme_bw() +
  geom_vline(xintercept = 5, linetype = "dotted") +
  ggtitle("Wykres PCA")

loading_scores <- pca$rotation[, 1]
var_scores <- abs(loading_scores)
var_rank <- sort(var_scores, decreasing = TRUE)
names(var_rank[1:30]) # top30 w pc1

# ---------------------- Selekcja zmiennych ----------------------

task <- makeClassifTask(data = train[-2], target = "poor") # nie bierzemy pod uwage zmiennej hid

fv <- generateFilterValuesData(task, method = "FSelector_chi.squared")
plotFilterValues(fv, filter = "FSelector_chi.squared", n.show = 10)

fv <- generateFilterValuesData(task, method = "praznik_CMIM")
plotFilterValues(fv, filter = "praznik_CMIM", n.show = 10)

# ---------------------- Random Forest ----------------------

train_rf <- data.frame(poor = y_combined[1:6000], X_combined[1:6000, -1]) # wyrzucamy zmienna hid

rf <- randomForest(poor ~ ., data = train_rf, ntree = 1000)
pred <- predict(rf, newdata = X_combined[6001:9000, -1])
confusionMatrix(pred, as.factor(ifelse(y_val == "Poor", "Poor", "Non-poor")), positive = "Poor")

# Tuningowanie parametrow
plot(rf)
e <- data.frame(rf$err.rate)
e[e$OOB == min(e$OOB), ]
bestmtry <- tuneRF(train_rf[-1], train_rf$poor, ntreeTry = 700)

rf <- randomForest(poor ~ ., data = train_rf, ntree = 700, mtry = 36, proximity = TRUE)
pred <- predict(rf, newdata = X_combined[6001:9000, -1])
confusionMatrix(pred, as.factor(ifelse(y_val == "Poor", "Poor", "Non-poor")), positive = "Poor")

y_val_pred <- predict(rf, newdata = X_combined[6001:9000, -1], type = "prob")[, 2]
write.table(y_val_pred, file = "test_rf.txt", row.names = FALSE, col.names = FALSE)

# # MDS plot
# distance.matrix <- as.dist(1 - rf$proximity) # domyslna metoda: odleglosc euklidesowa -> MDS <=> PCA
# mds.stuff <- cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)
# mds.var.per <- round(mds.stuff$eig / sum(mds.stuff$eig) * 100, 2)
# mds.values <- mds.stuff$points
# mds.data <- data.frame(Sample = rownames(mds.values),
#                        X = mds.values[, 1],
#                        Y = mds.values[, 2],
#                        Status = train$poor)
# ggplot(data = mds.data, aes(X, Y, label = Sample)) +
#   geom_point(aes(color = Status)) +
#   theme_bw() +
#   xlab(paste("MDS1 - ", mds.var.per[1], "%", sep = "")) +
#   ylab(paste("MDS2 - ", mds.var.per[2], "%", sep = "")) +
#   ggtitle("MDS plot using Random Forest Proximities") +
#   geom_vline(xintercept = -0.04, linetype = "dotted")

# ---------------------- XGBoost (model wybrany) ----------------------

y_val_xgb <- as.numeric(ifelse(y_val == "Poor", 1, 0))
y_train_xgb <- as.numeric(ifelse(y_train == "Poor", 1, 0))

X_train_xgb <- X_combined_numeric[1:6000, ]
X_val_xgb <- X_combined_numeric[6001:9000, ]

train_matrix <- xgb.DMatrix(data = as.matrix(X_train_xgb), label = y_train_xgb)
validation_matrix <- xgb.DMatrix(data = as.matrix(X_val_xgb), label = y_val_xgb)

params <- list("objective" = "binary:logistic", "eval_metric" = "auc")
watchlist <- list(train = train_matrix, validation = validation_matrix)

# xgb.fit <- xgb.train(params = params,
#                      data = train_matrix,
#                      watchlist = watchlist,
#                      nrounds = 645,
#                      eta = 0.045,
#                      max_depth = 3,
#                      gamma = 0)

xgb.fit <- xgb.train(params = params,
                     data = train_matrix,
                     watchlist = watchlist,
                     nrounds = 1000,
                     eta = 0.05,
                     max_depth = 3,
                     gamma = 1,
                     subsample = 0.6)

e <- data.frame(xgb.fit$evaluation_log)
plot(e$iter, e$train_auc, col = "blue", type = "l")
lines(e$iter, e$validation_auc, col = "red")
e[e$validation_auc == max(e$validation_auc), ]

e %>%
  ggplot(aes(iter, train_auc)) +
  geom_line(col = "blue") +
  geom_line(aes(iter, validation_auc), col = "red") +
  theme_bw() +
  labs(x = "Iteracja", y = "AUC", title = "Dopasowanie modelu XGBoost na podstawie AUC")

# Istotnosc zmiennych
importance_matrix <- xgb.importance(model = xgb.fit)
xgb.plot.importance(importance_matrix, top_n = 20, measure = "Gain")

y_val_pred <- predict(xgb.fit, newdata = as.matrix(X_val_xgb))
pred <- as.factor(ifelse(y_val_pred > 0.5, "Poor", "Non-poor"))
confusionMatrix(pred, as.factor(ifelse(y_val == "Poor", "Poor", "Non-poor")), positive = "Poor")

write.table(y_val_pred, file = "xgb.txt", row.names = FALSE, col.names = FALSE)

# Predykcja dla danych testowych
X_test_xgb <- X_combined_numeric[9001:12000, ]
y_test_pred <- predict(xgb.fit, newdata = as.matrix(X_test_xgb))
write.table(y_test_pred, file = "MWI.txt", row.names = FALSE, col.names = FALSE)

# ---------------------- SVM ----------------------

svm <- svm(poor ~ ., data = train_rf, probability = TRUE, kernel = "linear")
p <- attr(predict(svm, newdata = X_combined[6001:9000, -1], probability = TRUE), "probabilities")
pred <- as.factor(ifelse(p[, 2] > 0.5, "Poor", "Non-poor"))
confusionMatrix(pred, as.factor(ifelse(y_val == "Poor", "Poor", "Non-poor")), positive = "Poor")
y_val_pred <- p[, 2]
write.table(y_val_pred, file = "svm.txt", row.names = FALSE, col.names = FALSE)

# ---------------------- KNN ----------------------

pred_knn <- NULL
error_rate <- NULL

for(i in 1:10){
  pred_knn <- knn(train = X_train_xgb, test = X_val_xgb, cl = y_train_xgb, k = i, prob = TRUE)
  error_rate[i] <- mean(as.factor(y_val_xgb) != pred_knn)
}

k_values <- 1:10
error_df <- data.frame(error_rate, k_values)

error_df %>%
  ggplot(aes(x = k_values, y = error_rate)) +
  geom_point() +
  geom_line(lty = "dotted", col = "red") +
  theme_bw() +
  labs(x = "k", y = "B³¹d klasyfikacji", title = "B³¹d klasyfikacji na zbiorze walidacyjnym w zale¿noœci od liczby s¹siadów")

which.min(error_df$error_rate)

knn <- knn(train = X_train_xgb, test = X_val_xgb, cl = y_train_xgb, k = which.min(error_df$error_rate), prob = TRUE)
y_val_pred <- attr(knn, "prob")
confusionMatrix(knn, as.factor(y_val_xgb), positive = "1")
write.table(y_val_pred, file = "knn.txt", row.names = FALSE, col.names = FALSE) # k = 6

# BRUDNOPIS
# mean(train[train$poor == "Poor", "family_size"])
# mean(train[train$poor == "Non-poor", "family_size"])
# summary(train[train$poor == "Poor", c("cons_0801", "cons_0901")])
# summary(train[train$poor == "Non-poor", c("cons_0801", "cons_0901")])