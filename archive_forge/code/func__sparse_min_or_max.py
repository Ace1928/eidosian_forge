import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _sparse_min_or_max(X, axis, min_or_max):
    if axis is None:
        if 0 in X.shape:
            raise ValueError('zero-size array to reduction operation')
        zero = X.dtype.type(0)
        if X.nnz == 0:
            return zero
        m = min_or_max.reduce(X.data.ravel())
        if X.nnz != np.prod(X.shape):
            m = min_or_max(zero, m)
        return m
    if axis < 0:
        axis += 2
    if axis == 0 or axis == 1:
        return _min_or_max_axis(X, axis, min_or_max)
    else:
        raise ValueError('invalid axis, use 0 for rows, or 1 for columns')