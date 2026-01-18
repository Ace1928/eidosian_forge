from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def _check_fill_diagonal(arr, val):
    for wrap in (None, True, False):
        a = arr.copy()
        b = arr.copy()
        if wrap is None:
            params = {}
        else:
            params = {'wrap': wrap}
        pyfunc(a, val, **params)
        cfunc(b, val, **params)
        self.assertPreciseEqual(a, b)