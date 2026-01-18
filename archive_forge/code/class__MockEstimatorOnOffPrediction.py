import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils._metadata_requests import RequestMethod
from .metaestimators import available_if
from .validation import _check_sample_weight, _num_samples, check_array, check_is_fitted
class _MockEstimatorOnOffPrediction(BaseEstimator):
    """Estimator for which we can turn on/off the prediction methods.

    Parameters
    ----------
    response_methods: list of             {"predict", "predict_proba", "decision_function"}, default=None
        List containing the response implemented by the estimator. When, the
        response is in the list, it will return the name of the response method
        when called. Otherwise, an `AttributeError` is raised. It allows to
        use `getattr` as any conventional estimator. By default, no response
        methods are mocked.
    """

    def __init__(self, response_methods=None):
        self.response_methods = response_methods

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    @available_if(_check_response('predict'))
    def predict(self, X):
        return 'predict'

    @available_if(_check_response('predict_proba'))
    def predict_proba(self, X):
        return 'predict_proba'

    @available_if(_check_response('decision_function'))
    def decision_function(self, X):
        return 'decision_function'