import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import logsumexp
from .base import BaseEstimator, ClassifierMixin, _fit_context
from .preprocessing import LabelBinarizer, binarize, label_binarize
from .utils._param_validation import Interval
from .utils.extmath import safe_sparse_dot
from .utils.multiclass import _check_partial_fit_first_call
from .utils.validation import _check_sample_weight, check_is_fitted, check_non_negative
def _init_counters(self, n_classes, n_features):
    self.class_count_ = np.zeros(n_classes, dtype=np.float64)
    self.category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]