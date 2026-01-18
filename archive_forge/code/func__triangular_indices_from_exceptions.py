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
def _triangular_indices_from_exceptions(self, pyfunc, test_k=True):
    cfunc = jit(nopython=True)(pyfunc)
    for ndims in [0, 1, 3]:
        a = np.ones([5] * ndims)
        with self.assertTypingError() as raises:
            cfunc(a)
        self.assertIn('input array must be 2-d', str(raises.exception))
    if test_k:
        a = np.ones([5, 5])
        with self.assertTypingError() as raises:
            cfunc(a, k=0.5)
        self.assertIn('k must be an integer', str(raises.exception))