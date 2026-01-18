from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
class TestListOfList(ManagedListTestCase):

    def compile_and_test(self, pyfunc, *args):
        from copy import deepcopy
        expect_args = deepcopy(args)
        expect = pyfunc(*expect_args)
        njit_args = deepcopy(args)
        cfunc = jit(nopython=True)(pyfunc)
        got = cfunc(*njit_args)
        self.assert_list_element_precise_equal(expect=expect, got=got)
        self.assert_list_element_precise_equal(expect=expect_args, got=njit_args)

    def test_returning_list_of_list(self):

        def pyfunc():
            a = [[np.arange(i)] for i in range(4)]
            return a
        self.compile_and_test(pyfunc)

    @expect_reflection_failure
    def test_heterogeneous_list_error(self):

        def pyfunc(x):
            return x[1]
        cfunc = jit(nopython=True)(pyfunc)
        l2 = [[np.zeros(i) for i in range(5)], [np.ones(i) + 1j for i in range(5)]]
        l3 = [[np.zeros(i) for i in range(5)], [(1,)]]
        l4 = [[1], [{1}]]
        l5 = [[1], [{'a': 1}]]
        cfunc(l2)
        with self.assertRaises(TypeError) as raises:
            cfunc(l2)
        self.assertIn('reflected list(array(float64, 1d, C)) != reflected list(array(complex128, 1d, C))', str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            cfunc(l3)
        self.assertIn('reflected list(array(float64, 1d, C)) != reflected list((int64 x 1))', str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            cfunc(l4)
        self.assertIn('reflected list(int64) != reflected list(reflected set(int64))', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            cfunc(l5)
        self.assertIn("Cannot type list element of <class 'dict'>", str(raises.exception))

    @expect_reflection_failure
    def test_list_of_list_reflected(self):

        def pyfunc(l1, l2):
            l1.append(l2)
            l1[-1].append(123)
        cfunc = jit(nopython=True)(pyfunc)
        l1 = [[0, 1], [2, 3]]
        l2 = [4, 5]
        expect = (list(l1), list(l2))
        got = (list(l1), list(l2))
        pyfunc(*expect)
        cfunc(*got)
        self.assertEqual(expect, got)

    @expect_reflection_failure
    def test_heterogeneous_list(self):

        def pyfunc(x):
            return x[1]
        l1 = [[np.zeros(i) for i in range(5)], [np.ones(i) for i in range(5)]]
        cfunc = jit(nopython=True)(pyfunc)
        l1_got = cfunc(l1)
        self.assertPreciseEqual(pyfunc(l1), l1_got)

    @expect_reflection_failure
    def test_c01(self):

        def bar(x):
            return x.pop()
        r = [[np.zeros(0)], [np.zeros(10) * 1j]]
        self.compile_and_test(bar, r)
        with self.assertRaises(TypeError) as raises:
            self.compile_and_test(bar, r)
        self.assertIn('reflected list(array(float64, 1d, C)) != reflected list(array(complex128, 1d, C))', str(raises.exception))

    def test_c02(self):

        def bar(x):
            x.append(x)
            return x
        r = [[np.zeros(0)]]
        with self.assertRaises(errors.TypingError) as raises:
            self.compile_and_test(bar, r)
        self.assertIn('Invalid use of BoundFunction(list.append', str(raises.exception))

    def test_c03(self):

        def bar(x):
            f = x
            f[0] = 1
            return f
        r = [[np.arange(3)]]
        with self.assertRaises(errors.TypingError) as raises:
            self.compile_and_test(bar, r)
        self.assertIn('invalid setitem with value of {} to element of {}'.format(typeof(1), typeof(r[0])), str(raises.exception))

    def test_c04(self):

        def bar(x):
            f = x
            f[0][0] = 10
            return f
        r = [[np.arange(3)]]
        with self.assertRaises(errors.TypingError) as raises:
            self.compile_and_test(bar, r)
        self.assertIn('invalid setitem with value of {} to element of {}'.format(typeof(10), typeof(r[0][0])), str(raises.exception))

    @expect_reflection_failure
    def test_c05(self):

        def bar(x):
            f = x
            f[0][0] = np.array([x for x in np.arange(10).astype(np.intp)])
            return f
        r = [[np.arange(3).astype(np.intp)]]
        self.compile_and_test(bar, r)

    def test_c06(self):

        def bar(x):
            f = x
            f[0][0] = np.array([x + 1j for x in np.arange(10)])
            return f
        r = [[np.arange(3)]]
        with self.assertRaises(errors.TypingError) as raises:
            self.compile_and_test(bar, r)
        self.assertIn('invalid setitem with value', str(raises.exception))

    @expect_reflection_failure
    def test_c07(self):
        self.disable_leak_check()

        def bar(x):
            return x[-7]
        r = [[np.arange(3)]]
        cfunc = jit(nopython=True)(bar)
        with self.assertRaises(IndexError) as raises:
            cfunc(r)
        self.assertIn('getitem out of range', str(raises.exception))

    def test_c08(self):
        self.disable_leak_check()

        def bar(x):
            x[5] = 7
            return x
        r = [1, 2, 3]
        cfunc = jit(nopython=True)(bar)
        with self.assertRaises(IndexError) as raises:
            cfunc(r)
        self.assertIn('setitem out of range', str(raises.exception))

    def test_c09(self):

        def bar(x):
            x[-2] = 7j
            return x
        r = [1, 2, 3]
        with self.assertRaises(errors.TypingError) as raises:
            self.compile_and_test(bar, r)
        self.assertIn('invalid setitem with value', str(raises.exception))

    @expect_reflection_failure
    def test_c10(self):

        def bar(x):
            x[0], x[1] = (x[1], x[0])
            return x
        r = [[1, 2, 3], [4, 5, 6]]
        self.compile_and_test(bar, r)

    @expect_reflection_failure
    def test_c11(self):

        def bar(x):
            x[:] = x[::-1]
            return x
        r = [[1, 2, 3], [4, 5, 6]]
        self.compile_and_test(bar, r)

    def test_c12(self):

        def bar(x):
            del x[-1]
            return x
        r = [x for x in range(10)]
        self.compile_and_test(bar, r)