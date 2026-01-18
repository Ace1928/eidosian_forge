import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
@cupy._util.memoize()
def _output_dtype(dtype, value_type):
    if value_type != 'R2C':
        if dtype in [np.float16, np.float32]:
            return np.complex64
        elif dtype not in [np.complex64, np.complex128]:
            return np.complex128
    elif dtype in [np.complex64, np.complex128]:
        return np.dtype(dtype.char.lower())
    elif dtype == np.float16:
        return np.float32
    elif dtype not in [np.float32, np.float64]:
        return np.float64
    return dtype