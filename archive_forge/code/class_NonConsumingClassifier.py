from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
class NonConsumingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier which accepts no metadata on any method."""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def partial_fit(self, X, y, classes=None):
        return self

    def decision_function(self, X):
        return self.predict(X)

    def predict(self, X):
        return np.ones(len(X))