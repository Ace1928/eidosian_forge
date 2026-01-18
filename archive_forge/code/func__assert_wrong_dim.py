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
def _assert_wrong_dim(self, rn, cfunc):
    self.assert_wrong_dimensions_1D(rn, cfunc, (np.array([[[1]]], dtype=np.float64), np.ones(1)), False)
    self.assert_wrong_dimensions_1D(rn, cfunc, (np.ones(1), np.array([[[1]]], dtype=np.float64)), False)