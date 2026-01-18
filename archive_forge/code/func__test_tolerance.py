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
def _test_tolerance(sparse_container):
    X = X_diabetes if sparse_container is None else sparse_container(X_diabetes)
    ridge = Ridge(tol=1e-05, fit_intercept=False)
    ridge.fit(X, y_diabetes)
    score = ridge.score(X, y_diabetes)
    ridge2 = Ridge(tol=0.001, fit_intercept=False)
    ridge2.fit(X, y_diabetes)
    score2 = ridge2.score(X, y_diabetes)
    assert score >= score2