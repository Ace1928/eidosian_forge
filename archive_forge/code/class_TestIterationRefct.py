import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
class TestIterationRefct(MemoryLeakMixin, TestCase):

    def test_zip_with_arrays(self):

        @njit
        def foo(sequence):
            c = 0
            for a, b in zip(range(len(sequence)), sequence):
                c += (a + 1) * b.sum()
            return
        sequence = [np.arange(1 + i) for i in range(10)]
        self.assertEqual(foo(sequence), foo.py_func(sequence))