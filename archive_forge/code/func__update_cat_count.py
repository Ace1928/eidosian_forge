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
def _update_cat_count(X_feature, Y, cat_count, n_classes):
    for j in range(n_classes):
        mask = Y[:, j].astype(bool)
        if Y.dtype.type == np.int64:
            weights = None
        else:
            weights = Y[mask, j]
        counts = np.bincount(X_feature[mask], weights=weights)
        indices = np.nonzero(counts)[0]
        cat_count[j, indices] += counts[indices]