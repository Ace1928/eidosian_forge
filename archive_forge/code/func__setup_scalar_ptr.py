import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def _setup_scalar_ptr(handle, a, dtype):
    a, a_ptr = _get_scalar_ptr(a, dtype)
    mode = cublas.getPointerMode(handle)
    if isinstance(a, cupy.ndarray):
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    return (a, a_ptr, mode)