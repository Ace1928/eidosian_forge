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
def _triangular_matrix_exceptions(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    a = np.ones((5, 6))
    with self.assertTypingError() as raises:
        cfunc(a, k=1.5)
        self.assertIn('k must be an integer', str(raises.exception))