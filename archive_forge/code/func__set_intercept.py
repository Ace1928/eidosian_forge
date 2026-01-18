import numbers
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from ..base import (
from ..utils import check_array, check_random_state
from ..utils._array_api import get_namespace
from ..utils._seq_dataset import (
from ..utils.extmath import safe_sparse_dot
from ..utils.parallel import Parallel, delayed
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import FLOAT_DTYPES, _check_sample_weight, check_is_fitted
def _set_intercept(self, X_offset, y_offset, X_scale):
    """Set the intercept_"""
    if self.fit_intercept:
        self.coef_ = np.divide(self.coef_, X_scale, dtype=X_scale.dtype)
        self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
    else:
        self.intercept_ = 0.0