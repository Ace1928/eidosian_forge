import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import (
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
from ..utils.validation import (
from ._base import BaseEnsemble
def _get_median_predict(self, X, limit):
    predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T
    sorted_idx = np.argsort(predictions, axis=1)
    weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
    median_idx = median_or_above.argmax(axis=1)
    median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]
    return predictions[np.arange(_num_samples(X)), median_estimators]