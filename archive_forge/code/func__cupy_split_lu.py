from warnings import warn
import numpy
import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray
def _cupy_split_lu(LU, order='C'):
    assert LU._f_contiguous
    m, n = LU.shape
    k = min(m, n)
    order = 'F' if order == 'F' else 'C'
    L = cupy.empty((m, k), order=order, dtype=LU.dtype)
    U = cupy.empty((k, n), order=order, dtype=LU.dtype)
    size = m * n
    _kernel_cupy_split_lu(LU, m, n, k, L._c_contiguous, L, U, size=size)
    return (L, U)