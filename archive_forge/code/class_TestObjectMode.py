import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
class TestObjectMode(TestCase):

    def test_complex_constant(self):
        pyfunc = complex_constant
        cfunc = jit((), forceobj=True)(pyfunc)
        self.assertPreciseEqual(pyfunc(12), cfunc(12))

    def test_long_constant(self):
        pyfunc = long_constant
        cfunc = jit((), forceobj=True)(pyfunc)
        self.assertPreciseEqual(pyfunc(12), cfunc(12))

    def test_loop_nest(self):
        """
        Test bug that decref the iterator early.
        If the bug occurs, a segfault should occur
        """
        pyfunc = loop_nest_3
        cfunc = jit((), forceobj=True)(pyfunc)
        self.assertEqual(pyfunc(5, 5), cfunc(5, 5))

        def bm_pyfunc():
            pyfunc(5, 5)

        def bm_cfunc():
            cfunc(5, 5)
        utils.benchmark(bm_pyfunc)
        utils.benchmark(bm_cfunc)

    def test_array_of_object(self):
        cfunc = jit(forceobj=True)(array_of_object)
        objarr = np.array([object()] * 10)
        self.assertIs(cfunc(objarr), objarr)

    def test_sequence_contains(self):
        """
        Test handling of the `in` comparison
        """

        @jit(forceobj=True)
        def foo(x, y):
            return x in y
        self.assertTrue(foo(1, [0, 1]))
        self.assertTrue(foo(0, [0, 1]))
        self.assertFalse(foo(2, [0, 1]))
        with self.assertRaises(TypeError) as raises:
            foo(None, None)
        self.assertIn('is not iterable', str(raises.exception))

    def test_delitem(self):
        pyfunc = delitem_usecase
        cfunc = jit((), forceobj=True)(pyfunc)
        l = [3, 4, 5]
        cfunc(l)
        self.assertPreciseEqual(l, [])
        with self.assertRaises(TypeError):
            cfunc(42)

    def test_starargs_non_tuple(self):

        def consumer(*x):
            return x

        @jit(forceobj=True)
        def foo(x):
            return consumer(*x)
        arg = 'ijo'
        got = foo(arg)
        expect = foo.py_func(arg)
        self.assertEqual(got, tuple(arg))
        self.assertEqual(got, expect)

    def test_expr_undef(self):

        @jit(forceobj=True)
        def foo():
            return [x for x in (1, 2)]
        self.assertEqual(foo(), foo.py_func())