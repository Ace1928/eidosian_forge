import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _minor_reduce(X, ufunc):
    major_index = np.flatnonzero(np.diff(X.indptr))
    X = type(X)((X.data, X.indices, X.indptr), shape=X.shape)
    value = ufunc.reduceat(X.data, X.indptr[major_index])
    return (major_index, value)