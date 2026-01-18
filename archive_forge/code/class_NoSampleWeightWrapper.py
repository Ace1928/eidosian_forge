import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils._metadata_requests import RequestMethod
from .metaestimators import available_if
from .validation import _check_sample_weight, _num_samples, check_array, check_is_fitted
class NoSampleWeightWrapper(BaseEstimator):
    """Wrap estimator which will not expose `sample_weight`.

    Parameters
    ----------
    est : estimator, default=None
        The estimator to wrap.
    """

    def __init__(self, est=None):
        self.est = est

    def fit(self, X, y):
        return self.est.fit(X, y)

    def predict(self, X):
        return self.est.predict(X)

    def predict_proba(self, X):
        return self.est.predict_proba(X)

    def _more_tags(self):
        return {'_skip_test': True}