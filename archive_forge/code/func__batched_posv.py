import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusolver as _cusolver
from cupy.cuda import device as _device
import cupyx.cusolver
def _batched_posv(a, b):
    if not cupyx.cusolver.check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')
    dtype = _numpy.promote_types(a.dtype, b.dtype)
    dtype = _numpy.promote_types(dtype, 'f')
    if dtype == 'f':
        potrfBatched = _cusolver.spotrfBatched
        potrsBatched = _cusolver.spotrsBatched
    elif dtype == 'd':
        potrfBatched = _cusolver.dpotrfBatched
        potrsBatched = _cusolver.dpotrsBatched
    elif dtype == 'F':
        potrfBatched = _cusolver.cpotrfBatched
        potrsBatched = _cusolver.cpotrsBatched
    elif dtype == 'D':
        potrfBatched = _cusolver.zpotrfBatched
        potrsBatched = _cusolver.zpotrsBatched
    else:
        msg = 'dtype must be float32, float64, complex64 or complex128 (actual: {})'.format(a.dtype)
        raise ValueError(msg)
    a = a.astype(dtype, order='C', copy=True)
    ap = _cupy._core._mat_ptrs(a)
    lda, n = a.shape[-2:]
    batch_size = int(_numpy.prod(a.shape[:-2]))
    handle = _device.get_cusolver_handle()
    uplo = _cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = _cupy.empty(batch_size, dtype=_numpy.int32)
    potrfBatched(handle, uplo, n, ap.data.ptr, lda, dev_info.data.ptr, batch_size)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(potrfBatched, dev_info)
    b_shape = b.shape
    b = b.conj().reshape(batch_size, n, -1).astype(dtype, order='C', copy=True)
    bp = _cupy._core._mat_ptrs(b)
    ldb, nrhs = b.shape[-2:]
    dev_info = _cupy.empty(1, dtype=_numpy.int32)
    potrsBatched(handle, uplo, n, nrhs, ap.data.ptr, lda, bp.data.ptr, ldb, dev_info.data.ptr, batch_size)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(potrsBatched, dev_info)
    return b.conj().reshape(b_shape)