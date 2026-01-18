import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
def __kernel(a, neighborhood):
    self.check_stencil_arrays(a, neighborhood=neighborhood)
    __retdtype = kernel(a)
    __b0 = np.full(a.shape, cval, dtype=type(__retdtype))
    for __bn in range(1, a.shape[1] - 1):
        for __an in range(1, a.shape[0] - 1):
            __b0[__an, __bn] = a[__an + 0, __bn + 0]
    return __b0