import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal
def _replace_nan(a, val, out=None):
    if out is None or a.dtype != out.dtype:
        out = cupy.empty_like(a)
    _replace_nan_kernel(a, val, out)
    return out