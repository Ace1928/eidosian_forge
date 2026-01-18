import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestNamedTupleNRT(TestCase, MemoryLeakMixin):

    def test_return(self):
        pyfunc = make_point_nrt
        cfunc = jit(nopython=True)(pyfunc)
        for arg in (3, 0):
            expected = pyfunc(arg)
            got = cfunc(arg)
            self.assertIs(type(got), type(expected))
            self.assertPreciseEqual(got, expected)