import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _min_or_max_axis(X, axis, min_or_max):
    N = X.shape[axis]
    if N == 0:
        raise ValueError('zero-size array to reduction operation')
    M = X.shape[1 - axis]
    mat = X.tocsc() if axis == 0 else X.tocsr()
    mat.sum_duplicates()
    major_index, value = _minor_reduce(mat, min_or_max)
    not_full = np.diff(mat.indptr)[major_index] < N
    value[not_full] = min_or_max(value[not_full], 0)
    mask = value != 0
    major_index = np.compress(mask, major_index)
    value = np.compress(mask, value)
    if axis == 0:
        res = scipy.sparse.coo_matrix((value, (np.zeros(len(value)), major_index)), dtype=X.dtype, shape=(1, M))
    else:
        res = scipy.sparse.coo_matrix((value, (major_index, np.zeros(len(value)))), dtype=X.dtype, shape=(M, 1))
    return res.A.ravel()