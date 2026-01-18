import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
class TestCallError(unittest.TestCase):

    def test_readonly_array(self):

        @jit('(f8[:],)', nopython=True)
        def inner(x):
            return x

        @jit(nopython=True)
        def outer():
            return inner(gvalues)
        gvalues = np.ones(10, dtype=np.float64)
        with self.assertRaises(TypingError) as raises:
            outer()
        got = str(raises.exception)
        pat = 'Invalid use of.*readonly array\\(float64, 1d, C\\)'
        self.assertIsNotNone(re.search(pat, got))