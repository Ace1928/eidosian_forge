import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestOperations(TestCase):

    def test_len(self):
        pyfunc = len_usecase
        cfunc = njit((types.Tuple((types.int64, types.float32)),))(pyfunc)
        self.assertPreciseEqual(cfunc((4, 5)), 2)
        cfunc = njit((types.UniTuple(types.int64, 3),))(pyfunc)
        self.assertPreciseEqual(cfunc((4, 5, 6)), 3)

    def test_index_literal(self):

        def pyfunc(tup, idx):
            idx = literally(idx)
            return tup[idx]
        cfunc = njit(pyfunc)
        tup = (4, 3.1, 'sss')
        for i in range(len(tup)):
            self.assertPreciseEqual(cfunc(tup, i), tup[i])

    def test_index(self):
        pyfunc = tuple_index
        cfunc = njit((types.UniTuple(types.int64, 3), types.int64))(pyfunc)
        tup = (4, 3, 6)
        for i in range(len(tup)):
            self.assertPreciseEqual(cfunc(tup, i), tup[i])
        for i in range(len(tup) + 1):
            self.assertPreciseEqual(cfunc(tup, -i), tup[-i])
        with self.assertRaises(IndexError) as raises:
            cfunc(tup, len(tup))
        self.assertEqual('tuple index out of range', str(raises.exception))
        with self.assertRaises(IndexError) as raises:
            cfunc(tup, -(len(tup) + 1))
        self.assertEqual('tuple index out of range', str(raises.exception))
        args = (types.UniTuple(types.int64, 0), types.int64)
        cr = njit(args)(pyfunc).overloads[args]
        with self.assertRaises(IndexError) as raises:
            cr.entry_point((), 0)
        self.assertEqual('tuple index out of range', str(raises.exception))
        cfunc = njit((types.UniTuple(types.int64, 3), types.uintp))(pyfunc)
        for i in range(len(tup)):
            self.assertPreciseEqual(cfunc(tup, types.uintp(i)), tup[i])
        pyfunc = tuple_index_static
        for typ in (types.UniTuple(types.int64, 4), types.Tuple((types.int64, types.int32, types.int64, types.int32))):
            cfunc = njit((typ,))(pyfunc)
            tup = (4, 3, 42, 6)
            self.assertPreciseEqual(cfunc(tup), pyfunc(tup))
        typ = types.UniTuple(types.int64, 1)
        with self.assertTypingError():
            njit((typ,))(pyfunc)
        pyfunc = tuple_unpack_static_getitem_err
        with self.assertTypingError() as raises:
            njit(())(pyfunc)
        msg = "Cannot infer the type of variable 'c', have imprecise type: list(undefined)<iv=None>."
        self.assertIn(msg, str(raises.exception))

    def test_in(self):
        pyfunc = in_usecase
        cfunc = njit((types.int64, types.UniTuple(types.int64, 3)))(pyfunc)
        tup = (4, 1, 5)
        for i in range(5):
            self.assertPreciseEqual(cfunc(i, tup), pyfunc(i, tup))
        cfunc = njit((types.int64, types.Tuple([])))(pyfunc)
        self.assertPreciseEqual(cfunc(1, ()), pyfunc(1, ()))

    def check_slice(self, pyfunc):
        tup = (4, 5, 6, 7)
        cfunc = njit((types.UniTuple(types.int64, 4),))(pyfunc)
        self.assertPreciseEqual(cfunc(tup), pyfunc(tup))
        args = types.Tuple((types.int64, types.int32, types.int64, types.int32))
        cfunc = njit((args,))(pyfunc)
        self.assertPreciseEqual(cfunc(tup), pyfunc(tup))

    def test_slice2(self):
        self.check_slice(tuple_slice2)

    def test_slice3(self):
        self.check_slice(tuple_slice3)

    def test_bool(self):
        pyfunc = bool_usecase
        cfunc = njit((types.Tuple((types.int64, types.int32)),))(pyfunc)
        args = ((4, 5),)
        self.assertPreciseEqual(cfunc(*args), pyfunc(*args))
        cfunc = njit((types.UniTuple(types.int64, 3),))(pyfunc)
        args = ((4, 5, 6),)
        self.assertPreciseEqual(cfunc(*args), pyfunc(*args))
        cfunc = njit((types.Tuple(()),))(pyfunc)
        self.assertPreciseEqual(cfunc(()), pyfunc(()))

    def test_add(self):
        pyfunc = add_usecase
        samples = [(types.Tuple(()), ()), (types.UniTuple(types.int32, 0), ()), (types.UniTuple(types.int32, 1), (42,)), (types.Tuple((types.int64, types.float32)), (3, 4.5))]
        for (ta, a), (tb, b) in itertools.product(samples, samples):
            cfunc = njit((ta, tb))(pyfunc)
            expected = pyfunc(a, b)
            got = cfunc(a, b)
            self.assertPreciseEqual(got, expected, msg=(ta, tb))

    def _test_compare(self, pyfunc):

        def eq(pyfunc, cfunc, args):
            self.assertIs(cfunc(*args), pyfunc(*args), 'mismatch for arguments %s' % (args,))
        argtypes = [types.Tuple((types.int64, types.float32)), types.UniTuple(types.int32, 2)]
        for ta, tb in itertools.product(argtypes, argtypes):
            cfunc = njit((ta, tb))(pyfunc)
            for args in [((4, 5), (4, 5)), ((4, 5), (4, 6)), ((4, 6), (4, 5)), ((4, 5), (5, 4))]:
                eq(pyfunc, cfunc, args)
        argtypes = [types.Tuple((types.int64, types.float32)), types.UniTuple(types.int32, 3)]
        cfunc = njit(tuple(argtypes))(pyfunc)
        for args in [((4, 5), (4, 5, 6)), ((4, 5), (4, 4, 6)), ((4, 5), (4, 6, 7))]:
            eq(pyfunc, cfunc, args)

    def test_eq(self):
        self._test_compare(eq_usecase)

    def test_ne(self):
        self._test_compare(ne_usecase)

    def test_gt(self):
        self._test_compare(gt_usecase)

    def test_ge(self):
        self._test_compare(ge_usecase)

    def test_lt(self):
        self._test_compare(lt_usecase)

    def test_le(self):
        self._test_compare(le_usecase)