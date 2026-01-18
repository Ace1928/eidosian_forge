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
def _test_ridge_loo(sparse_container):
    n_samples = X_diabetes.shape[0]
    ret = []
    if sparse_container is None:
        X, fit_intercept = (X_diabetes, True)
    else:
        X, fit_intercept = (sparse_container(X_diabetes), False)
    ridge_gcv = _RidgeGCV(fit_intercept=fit_intercept)
    ridge_gcv.fit(X, y_diabetes)
    alpha_ = ridge_gcv.alpha_
    ret.append(alpha_)
    f = ignore_warnings
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    ridge_gcv2 = RidgeCV(fit_intercept=False, scoring=scoring)
    f(ridge_gcv2.fit)(X, y_diabetes)
    assert ridge_gcv2.alpha_ == pytest.approx(alpha_)

    def func(x, y):
        return -mean_squared_error(x, y)
    scoring = make_scorer(func)
    ridge_gcv3 = RidgeCV(fit_intercept=False, scoring=scoring)
    f(ridge_gcv3.fit)(X, y_diabetes)
    assert ridge_gcv3.alpha_ == pytest.approx(alpha_)
    scorer = get_scorer('neg_mean_squared_error')
    ridge_gcv4 = RidgeCV(fit_intercept=False, scoring=scorer)
    ridge_gcv4.fit(X, y_diabetes)
    assert ridge_gcv4.alpha_ == pytest.approx(alpha_)
    if sparse_container is None:
        ridge_gcv.fit(X, y_diabetes, sample_weight=np.ones(n_samples))
        assert ridge_gcv.alpha_ == pytest.approx(alpha_)
    Y = np.vstack((y_diabetes, y_diabetes)).T
    ridge_gcv.fit(X, Y)
    Y_pred = ridge_gcv.predict(X)
    ridge_gcv.fit(X, y_diabetes)
    y_pred = ridge_gcv.predict(X)
    assert_allclose(np.vstack((y_pred, y_pred)).T, Y_pred, rtol=1e-05)
    return ret