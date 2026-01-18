import warnings
import numpy as np
from numpy import asarray_chkfinite
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs
def _iuc(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = y != 0
    out[~nonzero] = False
    out[nonzero] = abs(x[nonzero] / y[nonzero]) < 1.0
    return out