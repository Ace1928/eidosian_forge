import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy._core import internal
from cupy.cuda import device
from cupyx.cusolver import check_availability
from cupyx.cusolver import _gesvdj_batched, _gesvd_batched
from cupyx.cusolver import _geqrf_orgqr_batched
from cupy.linalg import _util
def _svd_batched(a, full_matrices, compute_uv):
    batch_shape = a.shape[:-2]
    batch_size = internal.prod(batch_shape)
    n, m = a.shape[-2:]
    dtype, uv_dtype = _util.linalg_common_type(a)
    s_dtype = uv_dtype.char.lower()
    if batch_size == 0:
        k = min(m, n)
        s = cupy.empty(batch_shape + (k,), s_dtype)
        if compute_uv:
            if full_matrices:
                u = cupy.empty(batch_shape + (n, n), dtype=uv_dtype)
                vt = cupy.empty(batch_shape + (m, m), dtype=uv_dtype)
            else:
                u = cupy.empty(batch_shape + (n, k), dtype=uv_dtype)
                vt = cupy.empty(batch_shape + (k, m), dtype=uv_dtype)
            return (u, s, vt)
        else:
            return s
    elif m == 0 or n == 0:
        s = cupy.empty(batch_shape + (0,), s_dtype)
        if compute_uv:
            if full_matrices:
                u = _util.stacked_identity(batch_shape, n, uv_dtype)
                vt = _util.stacked_identity(batch_shape, m, uv_dtype)
            else:
                u = cupy.empty(batch_shape + (n, 0), dtype=uv_dtype)
                vt = cupy.empty(batch_shape + (0, m), dtype=uv_dtype)
            return (u, s, vt)
        else:
            return s
    a = a.reshape(-1, *a.shape[-2:])
    if runtime.is_hip or (m <= 32 and n <= 32):
        a = a.astype(dtype, order='C', copy=False)
        out = _gesvdj_batched(a, full_matrices, compute_uv, False)
    else:
        out = _gesvd_batched(a, dtype.char, full_matrices, compute_uv, False)
    if compute_uv:
        u, s, v = out
        u = u.astype(uv_dtype, copy=False)
        u = u.reshape(*batch_shape, *u.shape[-2:])
        s = s.astype(s_dtype, copy=False)
        s = s.reshape(*batch_shape, *s.shape[-1:])
        v = v.astype(uv_dtype, copy=False)
        v = v.reshape(*batch_shape, *v.shape[-2:])
        return (u, s, v.swapaxes(-2, -1).conj())
    else:
        s = out
        s = s.astype(s_dtype, copy=False)
        s = s.reshape(*batch_shape, *s.shape[-1:])
        return s