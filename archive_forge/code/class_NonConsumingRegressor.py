from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
class NonConsumingRegressor(RegressorMixin, BaseEstimator):
    """A classifier which accepts no metadata on any method."""

    def fit(self, X, y):
        return self

    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))