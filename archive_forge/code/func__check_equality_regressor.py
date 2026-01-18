import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def _check_equality_regressor(statistic, y_learn, y_pred_learn, y_test, y_pred_test):
    assert_array_almost_equal(np.tile(statistic, (y_learn.shape[0], 1)), y_pred_learn)
    assert_array_almost_equal(np.tile(statistic, (y_test.shape[0], 1)), y_pred_test)