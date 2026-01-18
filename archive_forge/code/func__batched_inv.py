import warnings
import numpy
from numpy import linalg
import cupy
from cupy._core import internal
from cupy.cuda import device
from cupy.linalg import _decomposition
from cupy.linalg import _util
from cupy.cublas import batched_gesv, get_batched_gesv_limit
import cupyx
def _batched_inv(a):
    dtype, out_dtype = _util.linalg_common_type(a)
    if a.size == 0:
        return cupy.empty(a.shape, out_dtype)
    if dtype == cupy.float32:
        getrf = cupy.cuda.cublas.sgetrfBatched
        getri = cupy.cuda.cublas.sgetriBatched
    elif dtype == cupy.float64:
        getrf = cupy.cuda.cublas.dgetrfBatched
        getri = cupy.cuda.cublas.dgetriBatched
    elif dtype == cupy.complex64:
        getrf = cupy.cuda.cublas.cgetrfBatched
        getri = cupy.cuda.cublas.cgetriBatched
    elif dtype == cupy.complex128:
        getrf = cupy.cuda.cublas.zgetrfBatched
        getri = cupy.cuda.cublas.zgetriBatched
    else:
        msg = 'dtype must be float32, float64, complex64 or complex128 (actual: {})'.format(a.dtype)
        raise ValueError(msg)
    if 0 in a.shape:
        return cupy.empty_like(a, dtype=out_dtype)
    a_shape = a.shape
    a = a.astype(dtype, order='C').reshape(-1, a_shape[-2], a_shape[-1])
    handle = device.get_cublas_handle()
    batch_size = a.shape[0]
    n = a.shape[1]
    lda = n
    step = n * lda * a.itemsize
    start = a.data.ptr
    stop = start + step * batch_size
    a_array = cupy.arange(start, stop, step, dtype=cupy.uintp)
    pivot_array = cupy.empty((batch_size, n), dtype=cupy.int32)
    info_array = cupy.empty((batch_size,), dtype=cupy.int32)
    getrf(handle, n, a_array.data.ptr, lda, pivot_array.data.ptr, info_array.data.ptr, batch_size)
    cupy.linalg._util._check_cublas_info_array_if_synchronization_allowed(getrf, info_array)
    c = cupy.empty_like(a)
    ldc = lda
    step = n * ldc * c.itemsize
    start = c.data.ptr
    stop = start + step * batch_size
    c_array = cupy.arange(start, stop, step, dtype=cupy.uintp)
    getri(handle, n, a_array.data.ptr, lda, pivot_array.data.ptr, c_array.data.ptr, ldc, info_array.data.ptr, batch_size)
    cupy.linalg._util._check_cublas_info_array_if_synchronization_allowed(getri, info_array)
    return c.reshape(a_shape).astype(out_dtype, copy=False)