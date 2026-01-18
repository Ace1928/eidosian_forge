import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def _get_scalar_ptr(a, dtype):
    if isinstance(a, cupy.ndarray):
        if a.dtype != dtype:
            a = cupy.array(a, dtype=dtype)
        a_ptr = a.data.ptr
    else:
        if not (isinstance(a, numpy.ndarray) and a.dtype == dtype):
            a = numpy.array(a, dtype=dtype)
        a_ptr = a.ctypes.data
    return (a, a_ptr)