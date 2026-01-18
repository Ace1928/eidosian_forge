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
def _update_sigma(self, X, alpha_, lambda_, keep_lambda):
    X_keep = X[:, keep_lambda]
    gram = np.dot(X_keep.T, X_keep)
    eye = np.eye(gram.shape[0], dtype=X.dtype)
    sigma_inv = lambda_[keep_lambda] * eye + alpha_ * gram
    sigma_ = pinvh(sigma_inv)
    return sigma_