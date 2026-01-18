import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _percentile(a, q, *, method='linear', **kwargs):
    return np.percentile(a, q, interpolation=method, **kwargs)