"""Training, cross-validation, and evaluation helpers.

Example contents:
- MODELS dictionary defining model pipelines and grids for hyperparameter tuning.
- `train_model(...)` to run CV and return the fitted search object.
- `evaluate(...)` to compute metrics on a holdout set.
"""

import random

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.data import test_train
SEED = 42


# simple linear regression
def linear_regression(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    X_train, X_test, y_train, y_test = test_train(SEED)

    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.0, 0.5, 1.0],
    }
    model = ElasticNet(max_iter=1000)

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    gscv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    gscv.fit(X_train, y_train)

    best = gscv.best_estimator_
    y_pred = best.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best alpha: {best['model'].alpha}")
    print(f"Best L1 ratio: {best['model'].l1_ratio}")
    print("=" * 30)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")

    return y_test, y_pred


# polynomial regression
def poly_regression(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    X_train, X_test, y_train, y_test = test_train(SEED)

    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.0, 0.5, 1.0],
    }
    model = ElasticNet(max_iter=1000)

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scaler", StandardScaler()), ("model", model)])

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    gscv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    gscv.fit(X_train, y_train)

    best = gscv.best_estimator_
    y_pred = best.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Best alpha: {best["model"].alpha}')
    print(f'Best L1 ratio: {best["model"].l1_ratio}')
    print('=' * 30)
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R2: {r2:.2f}')

    return y_test, y_pred



# KNN
def knn(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    X_train, X_test, y_train, y_test = test_train(SEED)

    param_grid = {
        "model__n_neighbors": [2, 5, 10, 20, 50],
        "model__p": [1, 2],
    }
    model = KNeighborsRegressor()

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    gscv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    gscv.fit(X_train, y_train)

    best = gscv.best_estimator_
    y_pred = best.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best n_neighbors: {best['model'].n_neighbors}")
    print(f"Best metric: {best['model'].p}")
    print("=" * 30)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")

    return y_test, y_pred

if __name__ == '__main__':
    print("\nLinear Regression Training:\n")
    linear_regression(SEED)

    print("\nPolynomial Regression Training:\n")
    poly_regression(SEED)

    print("\nKNN Training:\n")
    knn(SEED)
