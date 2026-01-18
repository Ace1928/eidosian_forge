import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _sparse_min_max(X, axis):
    return (_sparse_min_or_max(X, axis, np.minimum), _sparse_min_or_max(X, axis, np.maximum))