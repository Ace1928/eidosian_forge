import warnings
import numpy as np
from numpy import asarray_chkfinite
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs
def _rhp(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = y != 0
    out[~nonzero] = False
    out[nonzero] = np.real(x[nonzero] / y[nonzero]) > 0.0
    return out