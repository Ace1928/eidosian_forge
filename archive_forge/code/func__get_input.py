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
def _get_input(self, size1, size2, dtype):
    a = self._gen_input(size1, dtype, next(self.order1))
    b = self._gen_input(size2, dtype, next(self.order2))
    if np.iscomplexobj(a):
        b = b + 1j
    if np.iscomplexobj(b):
        a = a + 1j
    return (a, b)