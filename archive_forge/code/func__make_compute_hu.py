import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def _make_compute_hu(V):
    handle = device.get_cublas_handle()
    if V.dtype.char == 'f':
        gemv = _cublas.sgemv
    elif V.dtype.char == 'd':
        gemv = _cublas.dgemv
    elif V.dtype.char == 'F':
        gemv = _cublas.cgemv
    elif V.dtype.char == 'D':
        gemv = _cublas.zgemv
    n = V.shape[0]
    one = numpy.array(1.0, V.dtype)
    zero = numpy.array(0.0, V.dtype)
    mone = numpy.array(-1.0, V.dtype)

    def compute_hu(u, j):
        h = cupy.empty((j + 1,), dtype=V.dtype)
        gemv(handle, _cublas.CUBLAS_OP_C, n, j + 1, one.ctypes.data, V.data.ptr, n, u.data.ptr, 1, zero.ctypes.data, h.data.ptr, 1)
        gemv(handle, _cublas.CUBLAS_OP_N, n, j + 1, mone.ctypes.data, V.data.ptr, n, h.data.ptr, 1, one.ctypes.data, u.data.ptr, 1)
        return (h, u)
    return compute_hu