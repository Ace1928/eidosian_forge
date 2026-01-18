import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestNamedTuple(TestCase, MemoryLeakMixin):

    def test_unpack(self):

        def check(p):
            for pyfunc in (tuple_first, tuple_second):
                cfunc = jit(nopython=True)(pyfunc)
                self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check(Rect(4, 5))
        check(Rect(4, 5.5))

    def test_len(self):

        def check(p):
            pyfunc = len_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check(Rect(4, 5))
        check(Point(4, 5, 6))
        check(Rect(4, 5.5))
        check(Point(4, 5.5, 6j))

    def test_index(self):
        pyfunc = tuple_index
        cfunc = jit(nopython=True)(pyfunc)
        p = Point(4, 5, 6)
        for i in range(len(p)):
            self.assertPreciseEqual(cfunc(p, i), pyfunc(p, i))
        for i in range(len(p)):
            self.assertPreciseEqual(cfunc(p, types.uintp(i)), pyfunc(p, i))

    def test_bool(self):

        def check(p):
            pyfunc = bool_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check(Rect(4, 5))
        check(Rect(4, 5.5))
        check(Empty())

    def _test_compare(self, pyfunc):

        def eq(pyfunc, cfunc, args):
            self.assertIs(cfunc(*args), pyfunc(*args), 'mismatch for arguments %s' % (args,))
        cfunc = jit(nopython=True)(pyfunc)
        for a, b in [((4, 5), (4, 5)), ((4, 5), (4, 6)), ((4, 6), (4, 5)), ((4, 5), (5, 4))]:
            eq(pyfunc, cfunc, (Rect(*a), Rect(*b)))
        for a, b in [((4, 5), (4, 5, 6)), ((4, 5), (4, 4, 6)), ((4, 5), (4, 6, 7))]:
            eq(pyfunc, cfunc, (Rect(*a), Point(*b)))

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

    def test_getattr(self):
        pyfunc = getattr_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for args in ((4, 5, 6), (4, 5.5, 6j)):
            p = Point(*args)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

    def test_construct(self):

        def check(pyfunc):
            cfunc = jit(nopython=True)(pyfunc)
            for args in ((4, 5, 6), (4, 5.5, 6j)):
                expected = pyfunc(*args)
                got = cfunc(*args)
                self.assertIs(type(got), type(expected))
                self.assertPreciseEqual(got, expected)
        check(make_point)
        check(make_point_kws)

    def test_type(self):
        pyfunc = type_usecase
        cfunc = jit(nopython=True)(pyfunc)
        arg_tuples = [(4, 5, 6), (4, 5.5, 6j)]
        for tup_args, args in itertools.product(arg_tuples, arg_tuples):
            tup = Point(*tup_args)
            expected = pyfunc(tup, *args)
            got = cfunc(tup, *args)
            self.assertIs(type(got), type(expected))
            self.assertPreciseEqual(got, expected)

    def test_literal_unification(self):

        @jit(nopython=True)
        def Data1(value):
            return Rect(value, -321)

        @jit(nopython=True)
        def call(i, j):
            if j == 0:
                result = Data1(i)
            else:
                result = Rect(i, j)
            return result
        r = call(123, 1321)
        self.assertEqual(r, Rect(width=123, height=1321))
        r = call(123, 0)
        self.assertEqual(r, Rect(width=123, height=-321))

    def test_string_literal_in_ctor(self):

        @jit(nopython=True)
        def foo():
            return Rect(10, 'somestring')
        r = foo()
        self.assertEqual(r, Rect(width=10, height='somestring'))

    def test_dispatcher_mistreat(self):

        @jit(nopython=True)
        def foo(x):
            return x
        in1 = (1, 2, 3)
        out1 = foo(in1)
        self.assertEqual(in1, out1)
        in2 = Point(1, 2, 3)
        out2 = foo(in2)
        self.assertEqual(in2, out2)
        self.assertEqual(len(foo.nopython_signatures), 2)
        self.assertEqual(foo.nopython_signatures[0].args[0], typeof(in1))
        self.assertEqual(foo.nopython_signatures[1].args[0], typeof(in2))
        in3 = Point2(1, 2, 3)
        out3 = foo(in3)
        self.assertEqual(in3, out3)
        self.assertEqual(len(foo.nopython_signatures), 3)
        self.assertEqual(foo.nopython_signatures[2].args[0], typeof(in3))