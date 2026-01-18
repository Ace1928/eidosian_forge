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
def _check_array_split(self, func):
    pyfunc = func
    cfunc = jit(nopython=True)(pyfunc)

    def args_variations():
        yield (np.arange(8), 3)
        yield (list(np.arange(8)), 3)
        yield (tuple(np.arange(8)), 3)
        yield (np.arange(24).reshape(12, 2), 5)
    for args in args_variations():
        expected = pyfunc(*args)
        got = cfunc(*args)
        np.testing.assert_equal(expected, list(got))