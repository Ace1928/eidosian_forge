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
class TestNpArray(MemoryLeakMixin, BaseTest):

    def test_0d(self):

        def pyfunc(arg):
            return np.array(arg)
        cfunc = nrtjit(pyfunc)
        got = cfunc(42)
        self.assertPreciseEqual(got, np.array(42, dtype=np.intp))
        got = cfunc(2.5)
        self.assertPreciseEqual(got, np.array(2.5))

    def test_0d_with_dtype(self):

        def pyfunc(arg):
            return np.array(arg, dtype=np.int16)
        self.check_outputs(pyfunc, [(42,), (3.5,)])

    def test_1d(self):

        def pyfunc(arg):
            return np.array(arg)
        cfunc = nrtjit(pyfunc)
        got = cfunc([2, 3, 42])
        self.assertPreciseEqual(got, np.intp([2, 3, 42]))
        got = cfunc((1.0, 2.5j, 42))
        self.assertPreciseEqual(got, np.array([1.0, 2.5j, 42]))
        got = cfunc(())
        self.assertPreciseEqual(got, np.float64(()))

    def test_1d_with_dtype(self):

        def pyfunc(arg):
            return np.array(arg, dtype=np.float32)
        self.check_outputs(pyfunc, [([2, 42],), ([3.5, 1.0],), ((1, 3.5, 42),), ((),)])

    def test_1d_with_str_dtype(self):

        def pyfunc(arg):
            return np.array(arg, dtype='float32')
        self.check_outputs(pyfunc, [([2, 42],), ([3.5, 1.0],), ((1, 3.5, 42),), ((),)])

    def test_1d_with_non_const_str_dtype(self):

        @njit
        def func(arg, dt):
            return np.array(arg, dtype=dt)
        with self.assertRaises(TypingError) as raises:
            func((5, 3), 'int32')
        excstr = str(raises.exception)
        msg = f'If np.array dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_2d(self):

        def pyfunc(arg):
            return np.array(arg)
        cfunc = nrtjit(pyfunc)
        got = cfunc([(1, 2), (3, 4)])
        self.assertPreciseEqual(got, np.intp([[1, 2], [3, 4]]))
        got = cfunc([(1, 2.5), (3, 4.5)])
        self.assertPreciseEqual(got, np.float64([[1, 2.5], [3, 4.5]]))
        got = cfunc(([1, 2], [3, 4]))
        self.assertPreciseEqual(got, np.intp([[1, 2], [3, 4]]))
        got = cfunc(([1, 2], [3.5, 4.5]))
        self.assertPreciseEqual(got, np.float64([[1, 2], [3.5, 4.5]]))
        got = cfunc(((1.5, 2), (3.5, 4.5)))
        self.assertPreciseEqual(got, np.float64([[1.5, 2], [3.5, 4.5]]))
        got = cfunc(((), ()))
        self.assertPreciseEqual(got, np.float64(((), ())))

    def test_2d_with_dtype(self):

        def pyfunc(arg):
            return np.array(arg, dtype=np.int32)
        cfunc = nrtjit(pyfunc)
        got = cfunc([(1, 2.5), (3, 4.5)])
        self.assertPreciseEqual(got, np.int32([[1, 2], [3, 4]]))

    def test_raises(self):

        def pyfunc(arg):
            return np.array(arg)
        cfunc = nrtjit(pyfunc)

        @contextlib.contextmanager
        def check_raises(msg):
            with self.assertRaises(TypingError) as raises:
                yield
            self.assertIn(msg, str(raises.exception))
        with check_raises('array(float64, 1d, C) not allowed in a homogeneous sequence'):
            cfunc(np.array([1.0]))
        with check_raises('type Tuple(int64, reflected list(int64)<iv=None>) does not have a regular shape'):
            cfunc((np.int64(1), [np.int64(2)]))
        with check_raises('cannot convert Tuple(int64, Record(a[type=int32;offset=0],b[type=float32;offset=4];8;False)) to a homogeneous type'):
            st = np.dtype([('a', 'i4'), ('b', 'f4')])
            val = np.zeros(1, dtype=st)[0]
            cfunc(((1, 2), (np.int64(1), val)))

    def test_bad_array(self):

        @njit
        def func(obj):
            return np.array(obj)
        msg = '.*The argument "object" must be array-like.*'
        with self.assertRaisesRegex(TypingError, msg) as raises:
            func(None)

    def test_bad_dtype(self):

        @njit
        def func(obj, dt):
            return np.array(obj, dt)
        msg = '.*The argument "dtype" must be a data-type if it is provided.*'
        with self.assertRaisesRegex(TypingError, msg) as raises:
            func(5, 4)