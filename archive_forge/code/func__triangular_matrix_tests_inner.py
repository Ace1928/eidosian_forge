import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
@staticmethod
def _triangular_matrix_tests_inner(self, pyfunc, _check):

    def check_odd(a):
        _check(a)
        a = a.reshape((9, 7))
        _check(a)
        a = a.reshape((7, 1, 3, 3))
        _check(a)
        _check(a.T)

    def check_even(a):
        _check(a)
        a = a.reshape((4, 16))
        _check(a)
        a = a.reshape((4, 2, 2, 4))
        _check(a)
        _check(a.T)
    check_odd(np.arange(63) + 10.5)
    check_even(np.arange(64) - 10.5)
    _check(np.arange(360).reshape(3, 4, 5, 6))
    _check(np.array([]))
    _check(np.arange(9).reshape((3, 3))[::-1])
    _check(np.arange(9).reshape((3, 3), order='F'))
    arr = (np.arange(64) - 10.5).reshape((4, 2, 2, 4))
    _check(arr)
    _check(np.asfortranarray(arr))