import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def _setup_result_ptr(handle, out, dtype):
    mode = cublas.getPointerMode(handle)
    if out is None or isinstance(out, cupy.ndarray):
        if out is None or out.dtype != dtype:
            result = cupy.empty([], dtype=dtype)
        else:
            result = out
        result_ptr = result.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    elif isinstance(out, numpy.ndarray):
        if out.dtype != dtype:
            result = numpy.empty([], dtype=dtype)
        else:
            result = out
        result_ptr = result.ctypes.data
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    else:
        raise TypeError('out must be either cupy or numpy ndarray')
    return (result_ptr, result, mode)