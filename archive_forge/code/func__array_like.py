import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
def _array_like(x, x0):
    """Return ndarray `x` as same array subclass and shape as `x0`"""
    x = np.reshape(x, np.shape(x0))
    wrap = getattr(x0, '__array_wrap__', x.__array_wrap__)
    return wrap(x)