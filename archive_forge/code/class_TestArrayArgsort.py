import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
class TestArrayArgsort(MemoryLeakMixin, TestCase):
    """Tests specific to array.argsort"""

    def test_exceptions(self):

        @njit
        def nonliteral_kind(kind):
            np.arange(5).argsort(kind=kind)
        with self.assertRaises(errors.TypingError) as raises:
            nonliteral_kind('quicksort')
        expect = '"kind" must be a string literal'
        self.assertIn(expect, str(raises.exception))

        @njit
        def unsupported_kwarg():
            np.arange(5).argsort(foo='')
        with self.assertRaises(errors.TypingError) as raises:
            unsupported_kwarg()
        expect = "Unsupported keywords: ['foo']"
        self.assertIn(expect, str(raises.exception))