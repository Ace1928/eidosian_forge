from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
from scipy.linalg import norm
from scipy.sparse import dia_matrix, issparse
from scipy.sparse.linalg import eigsh, svds
from ..base import BaseEstimator, BiclusterMixin, _fit_context
from ..utils import check_random_state, check_scalar
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import make_nonnegative, randomized_svd, safe_sparse_dot
from ..utils.validation import assert_all_finite
from ._kmeans import KMeans, MiniBatchKMeans
def _log_normalize(X):
    """Normalize ``X`` according to Kluger's log-interactions scheme."""
    X = make_nonnegative(X, min_value=1)
    if issparse(X):
        raise ValueError('Cannot compute log of a sparse matrix, because log(x) diverges to -infinity as x goes to 0.')
    L = np.log(X)
    row_avg = L.mean(axis=1)[:, np.newaxis]
    col_avg = L.mean(axis=0)
    avg = L.mean()
    return L - row_avg - col_avg + avg