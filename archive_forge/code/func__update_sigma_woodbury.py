import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg import pinvh
from ..base import RegressorMixin, _fit_context
from ..utils import _safe_indexing
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import fast_logdet
from ..utils.validation import _check_sample_weight
from ._base import LinearModel, _preprocess_data, _rescale_data
def _update_sigma_woodbury(self, X, alpha_, lambda_, keep_lambda):
    n_samples = X.shape[0]
    X_keep = X[:, keep_lambda]
    inv_lambda = 1 / lambda_[keep_lambda].reshape(1, -1)
    sigma_ = pinvh(np.eye(n_samples, dtype=X.dtype) / alpha_ + np.dot(X_keep * inv_lambda, X_keep.T))
    sigma_ = np.dot(sigma_, X_keep * inv_lambda)
    sigma_ = -np.dot(inv_lambda.reshape(-1, 1) * X_keep.T, sigma_)
    sigma_[np.diag_indices(sigma_.shape[1])] += 1.0 / lambda_[keep_lambda]
    return sigma_