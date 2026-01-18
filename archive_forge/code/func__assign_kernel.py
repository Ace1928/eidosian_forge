import math
import functools
import operator
import copy
from ctypes import c_void_p
import numpy as np
import numba
from numba import _devicearray
from numba.cuda.cudadrv import devices
from numba.cuda.cudadrv import driver as _driver
from numba.core import types, config
from numba.np.unsafe.ndarray import to_fixed_tuple
from numba.misc import dummyarray
from numba.np import numpy_support
from numba.cuda.api_util import prepare_shape_strides_dtype
from numba.core.errors import NumbaPerformanceWarning
from warnings import warn
@lru_cache
def _assign_kernel(ndim):
    """
    A separate method so we don't need to compile code every assignment (!).

    :param ndim: We need to have static array sizes for cuda.local.array, so
        bake in the number of dimensions into the kernel
    """
    from numba import cuda
    if ndim == 0:

        @cuda.jit
        def kernel(lhs, rhs):
            lhs[()] = rhs[()]
        return kernel

    @cuda.jit
    def kernel(lhs, rhs):
        location = cuda.grid(1)
        n_elements = 1
        for i in range(lhs.ndim):
            n_elements *= lhs.shape[i]
        if location >= n_elements:
            return
        idx = cuda.local.array(shape=(2, ndim), dtype=types.int64)
        for i in range(ndim - 1, -1, -1):
            idx[0, i] = location % lhs.shape[i]
            idx[1, i] = location % lhs.shape[i] * (rhs.shape[i] > 1)
            location //= lhs.shape[i]
        lhs[to_fixed_tuple(idx[0], ndim)] = rhs[to_fixed_tuple(idx[1], ndim)]
    return kernel