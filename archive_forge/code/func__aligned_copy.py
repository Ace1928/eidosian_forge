import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
def _aligned_copy(self, arr):
    size = (arr.size + 1) * arr.itemsize + 1
    datasize = arr.size * arr.itemsize
    tmp = np.empty(size, dtype=np.uint8)
    for i in range(arr.itemsize + 1):
        new = tmp[i:i + datasize].view(dtype=arr.dtype)
        if new.flags.aligned:
            break
    else:
        raise Exception('Could not obtain aligned array')
    if arr.flags.c_contiguous:
        new = np.reshape(new, arr.shape, order='C')
    else:
        new = np.reshape(new, arr.shape, order='F')
    new[:] = arr[:]
    assert new.flags.aligned
    return new