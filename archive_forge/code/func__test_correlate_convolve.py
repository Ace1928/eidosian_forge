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
def _test_correlate_convolve(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    lengths = (1, 2, 3, 7)
    dts = [np.int8, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
    modes = ['full', 'valid', 'same']
    for dt1, dt2, n, m, mode in itertools.product(dts, dts, lengths, lengths, modes):
        a = np.arange(n, dtype=dt1)
        v = np.arange(m, dtype=dt2)
        if np.issubdtype(dt1, np.complexfloating):
            a = (a + 1j * a).astype(dt1)
        if np.issubdtype(dt2, np.complexfloating):
            v = (v + 1j * v).astype(dt2)
        expected = pyfunc(a, v, mode=mode)
        got = cfunc(a, v, mode=mode)
        self.assertPreciseEqual(expected, got)
    _a = np.arange(12).reshape(4, 3)
    _b = np.arange(12)
    for x, y in [(_a, _b), (_b, _a)]:
        with self.assertRaises(TypingError) as raises:
            cfunc(x, y)
        msg = 'only supported on 1D arrays'
        self.assertIn(msg, str(raises.exception))