import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
@unittest.skipUnless(hasattr(np, 'stack'), "this Numpy doesn't have np.stack()")
class TestNpStack(MemoryLeakMixin, TestCase):
    """
    Tests for np.stack().
    """

    def _3d_arrays(self):
        a = np.arange(24).reshape((4, 3, 2))
        b = a + 10
        c = (b + 10).copy(order='F')
        d = (c + 10)[::-1]
        e = (d + 10)[..., ::-1]
        return (a, b, c, d, e)

    @contextlib.contextmanager
    def assert_invalid_sizes(self):
        with self.assertRaises(ValueError) as raises:
            yield
        self.assertIn('all input arrays must have the same shape', str(raises.exception))

    def check_stack(self, pyfunc, cfunc, args):
        expected = pyfunc(*args)
        got = cfunc(*args)
        self.assertEqual(got.shape, expected.shape)
        self.assertPreciseEqual(got.flatten(), expected.flatten())

    def check_3d(self, pyfunc, cfunc, generate_starargs):

        def check(a, b, c, args):
            self.check_stack(pyfunc, cfunc, (a, b, c) + args)

        def check_all_axes(a, b, c):
            for args in generate_starargs():
                check(a, b, c, args)
        a, b, c, d, e = self._3d_arrays()
        check_all_axes(a, b, b)
        check_all_axes(a, b, c)
        check_all_axes(a.T, b.T, a.T)
        check_all_axes(a.T, b.T, c.T)
        check_all_axes(a.T, b.T, d.T)
        check_all_axes(d.T, e.T, d.T)
        check_all_axes(a, b.astype(np.float64), b)

    def check_runtime_errors(self, cfunc, generate_starargs):
        self.assert_no_memory_leak()
        self.disable_leak_check()
        a, b, c, d, e = self._3d_arrays()
        with self.assert_invalid_sizes():
            args = next(generate_starargs())
            cfunc(a[:-1], b, c, *args)

    def test_3d(self):
        """
        stack(3d arrays, axis)
        """
        pyfunc = np_stack2
        cfunc = nrtjit(pyfunc)

        def generate_starargs():
            for axis in range(3):
                yield (axis,)
                yield (-3 + axis,)
        self.check_3d(pyfunc, cfunc, generate_starargs)
        self.check_runtime_errors(cfunc, generate_starargs)

    def test_3d_no_axis(self):
        """
        stack(3d arrays)
        """
        pyfunc = np_stack1
        cfunc = nrtjit(pyfunc)

        def generate_starargs():
            yield ()
        self.check_3d(pyfunc, cfunc, generate_starargs)
        self.check_runtime_errors(cfunc, generate_starargs)

    def test_0d(self):
        """
        stack(0d arrays)
        """
        pyfunc = np_stack1
        cfunc = nrtjit(pyfunc)
        a = np.array(42)
        b = np.array(-5j)
        c = np.array(True)
        self.check_stack(pyfunc, cfunc, (a, b, c))

    def check_xxstack(self, pyfunc, cfunc):
        """
        3d and 0d tests for hstack(), vstack(), dstack().
        """

        def generate_starargs():
            yield ()
        self.check_3d(pyfunc, cfunc, generate_starargs)
        a = np.array(42)
        b = np.array(-5j)
        c = np.array(True)
        self.check_stack(pyfunc, cfunc, (a, b, a))

    def test_hstack(self):
        pyfunc = np_hstack
        cfunc = nrtjit(pyfunc)
        self.check_xxstack(pyfunc, cfunc)
        a = np.arange(5)
        b = np.arange(6) + 10
        self.check_stack(pyfunc, cfunc, (a, b, b))
        a = np.arange(6).reshape((2, 3))
        b = np.arange(8).reshape((2, 4)) + 100
        self.check_stack(pyfunc, cfunc, (a, b, a))

    def test_vstack(self):
        for pyfunc in (np_vstack, np_row_stack):
            cfunc = nrtjit(pyfunc)
            self.check_xxstack(pyfunc, cfunc)
            a = np.arange(5)
            b = a + 10
            self.check_stack(pyfunc, cfunc, (a, b, b))
            a = np.arange(6).reshape((3, 2))
            b = np.arange(8).reshape((4, 2)) + 100
            self.check_stack(pyfunc, cfunc, (a, b, b))

    def test_dstack(self):
        pyfunc = np_dstack
        cfunc = nrtjit(pyfunc)
        self.check_xxstack(pyfunc, cfunc)
        a = np.arange(5)
        b = a + 10
        self.check_stack(pyfunc, cfunc, (a, b, b))
        a = np.arange(12).reshape((3, 4))
        b = a + 100
        self.check_stack(pyfunc, cfunc, (a, b, b))

    def test_column_stack(self):
        pyfunc = np_column_stack
        cfunc = nrtjit(pyfunc)
        a = np.arange(4)
        b = a + 10
        c = np.arange(12).reshape((4, 3))
        self.check_stack(pyfunc, cfunc, (a, b, c))
        self.assert_no_memory_leak()
        self.disable_leak_check()
        a = np.array(42)
        with self.assertTypingError():
            cfunc((a, a, a))
        a = a.reshape((1, 1, 1))
        with self.assertTypingError():
            cfunc((a, a, a))

    def test_bad_arrays(self):
        for pyfunc in (np_stack1, np_hstack, np_vstack, np_dstack, np_column_stack):
            cfunc = nrtjit(pyfunc)
            c = np.arange(12).reshape((4, 3))
            with self.assertTypingError() as raises:
                cfunc(c, 1, c)
            self.assertIn('expecting a non-empty tuple of arrays', str(raises.exception))