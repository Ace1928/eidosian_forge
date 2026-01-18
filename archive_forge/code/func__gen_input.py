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
def _gen_input(self, size, dtype, order):
    if not isinstance(size, tuple):
        return size
    elif len(size) == 1:
        return self.sample_vector(size[0], dtype)
    else:
        return self.sample_vector(size[0] * size[1], dtype).reshape(size, order=order)