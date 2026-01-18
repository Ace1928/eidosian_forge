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
def _rescale_data(X, y, sample_weight, inplace=False):
    """Rescale data sample-wise by square root of sample_weight.

    For many linear models, this enables easy support for sample_weight because

        (y - X w)' S (y - X w)

    with S = diag(sample_weight) becomes

        ||y_rescaled - X_rescaled w||_2^2

    when setting

        y_rescaled = sqrt(S) y
        X_rescaled = sqrt(S) X

    Returns
    -------
    X_rescaled : {array-like, sparse matrix}

    y_rescaled : {array-like, sparse matrix}
    """
    n_samples = X.shape[0]
    sample_weight_sqrt = np.sqrt(sample_weight)
    if sp.issparse(X) or sp.issparse(y):
        sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    if sp.issparse(X):
        X = safe_sparse_dot(sw_matrix, X)
    elif inplace:
        X *= sample_weight_sqrt[:, np.newaxis]
    else:
        X = X * sample_weight_sqrt[:, np.newaxis]
    if sp.issparse(y):
        y = safe_sparse_dot(sw_matrix, y)
    elif inplace:
        if y.ndim == 1:
            y *= sample_weight_sqrt
        else:
            y *= sample_weight_sqrt[:, np.newaxis]
    elif y.ndim == 1:
        y = y * sample_weight_sqrt
    else:
        y = y * sample_weight_sqrt[:, np.newaxis]
    return (X, y, sample_weight_sqrt)