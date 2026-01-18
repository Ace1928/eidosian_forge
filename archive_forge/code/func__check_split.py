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
def _check_split(self, func):
    pyfunc = func
    cfunc = jit(nopython=True)(pyfunc)

    def args_variations():
        a = np.arange(100)
        yield (a, 2)
        yield (a, 2, 0)
        yield (a, [1, 4, 72])
        yield (list(a), [1, 4, 72])
        yield (tuple(a), [1, 4, 72])
        yield (a, [1, 4, 72], 0)
        yield (list(a), [1, 4, 72], 0)
        yield (tuple(a), [1, 4, 72], 0)
        a = np.arange(64).reshape(4, 4, 4)
        yield (a, 2)
        yield (a, 2, 0)
        yield (a, 2, 1)
        yield (a, [2, 1, 5])
        yield (a, [2, 1, 5], 1)
        yield (a, [2, 1, 5], 2)
        yield (a, [1, 3])
        yield (a, [1, 3], 1)
        yield (a, [1, 3], 2)
        yield (a, [1], -1)
        yield (a, [1], -2)
        yield (a, [1], -3)
        yield (a, np.array([], dtype=np.int64), 0)
        a = np.arange(100).reshape(2, -1)
        yield (a, 1)
        yield (a, 1, 0)
        yield (a, [1], 0)
        yield (a, 50, 1)
        yield (a, np.arange(10, 50, 10), 1)
        yield (a, (1,))
        yield (a, (np.int32(4), 10))
        a = np.array([])
        yield (a, 1)
        yield (a, 2)
        yield (a, (2, 3), 0)
        yield (a, 1, 0)
        a = np.array([[]])
        yield (a, 1)
        yield (a, (2, 3), 1)
        yield (a, 1, 0)
        yield (a, 1, 1)
    for args in args_variations():
        expected = pyfunc(*args)
        got = cfunc(*args)
        np.testing.assert_equal(expected, list(got))