import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestTupleReturn(TestCase):

    def test_array_tuple(self):
        aryty = types.Array(types.float64, 1, 'C')
        cfunc = njit((aryty, aryty))(tuple_return_usecase)
        a = b = np.arange(5, dtype='float64')
        ra, rb = cfunc(a, b)
        self.assertPreciseEqual(ra, a)
        self.assertPreciseEqual(rb, b)
        del a, b
        self.assertPreciseEqual(ra, rb)

    def test_scalar_tuple(self):
        scalarty = types.float32
        cfunc = njit((scalarty, scalarty))(tuple_return_usecase)
        a = b = 1
        ra, rb = cfunc(a, b)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)

    def test_hetero_tuple(self):
        alltypes = []
        allvalues = []
        alltypes.append((types.int32, types.int64))
        allvalues.append((1, 2))
        alltypes.append((types.float32, types.float64))
        allvalues.append((1.125, 0.25))
        alltypes.append((types.int32, types.float64))
        allvalues.append((1231, 0.5))
        for (ta, tb), (a, b) in zip(alltypes, allvalues):
            cfunc = njit((ta, tb))(tuple_return_usecase)
            ra, rb = cfunc(a, b)
            self.assertPreciseEqual((ra, rb), (a, b))