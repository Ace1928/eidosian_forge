import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def _test_ridge_cv(sparse_container):
    X = X_diabetes if sparse_container is None else sparse_container(X_diabetes)
    ridge_cv = RidgeCV()
    ridge_cv.fit(X, y_diabetes)
    ridge_cv.predict(X)
    assert len(ridge_cv.coef_.shape) == 1
    assert type(ridge_cv.intercept_) == np.float64
    cv = KFold(5)
    ridge_cv.set_params(cv=cv)
    ridge_cv.fit(X, y_diabetes)
    ridge_cv.predict(X)
    assert len(ridge_cv.coef_.shape) == 1
    assert type(ridge_cv.intercept_) == np.float64