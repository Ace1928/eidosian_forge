import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
class TestDUFuncMethods(TestCase):

    def _check_reduce(self, ufunc, dtype=None, initial=None):

        @njit
        def foo(a, axis, dtype, initial):
            return ufunc.reduce(a, axis=axis, dtype=dtype, initial=initial)
        inputs = [np.arange(5), np.arange(4).reshape(2, 2), np.arange(40).reshape(5, 4, 2)]
        for array in inputs:
            for axis in range(array.ndim):
                expected = foo.py_func(array, axis, dtype, initial)
                got = foo(array, axis, dtype, initial)
                self.assertPreciseEqual(expected, got)

    def _check_reduce_axis(self, ufunc, dtype, initial=None):

        @njit
        def foo(a, axis):
            return ufunc.reduce(a, axis=axis, initial=initial)

        def _check(*args):
            try:
                expected = foo.py_func(array, axis)
            except ValueError as e:
                self.assertEqual(e.args[0], exc_msg)
                with self.assertRaisesRegex(TypingError, exc_msg):
                    got = foo(array, axis)
            else:
                got = foo(array, axis)
                self.assertPreciseEqual(expected, got)
        exc_msg = f"reduction operation '{ufunc.__name__}' is not reorderable, so at most one axis may be specified"
        inputs = [np.arange(40, dtype=dtype).reshape(5, 4, 2), np.arange(10, dtype=dtype)]
        for array in inputs:
            for i in range(1, array.ndim + 1):
                for axis in itertools.combinations(range(array.ndim), r=i):
                    _check(array, axis)
            for axis in ((), None):
                _check(array, axis)

    def test_add_reduce(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd)
        self._check_reduce_axis(duadd, dtype=np.int64)

    def test_mul_reduce(self):
        dumul = vectorize('int64(int64, int64)', identity=1)(pymult)
        self._check_reduce(dumul)

    def test_non_associative_reduce(self):
        dusub = vectorize('int64(int64, int64)')(pysub)
        dudiv = vectorize('int64(int64, int64)')(pydiv)
        self._check_reduce(dusub)
        self._check_reduce_axis(dusub, dtype=np.int64)
        self._check_reduce(dudiv)
        self._check_reduce_axis(dudiv, dtype=np.int64)

    def test_reduce_dtype(self):
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.float64)

    def test_min_reduce(self):
        dumin = vectorize('int64(int64, int64)')(pymin)
        self._check_reduce(dumin, initial=10)
        self._check_reduce_axis(dumin, dtype=np.int64)

    def test_add_reduce_initial(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.int64, initial=100)

    def test_add_reduce_no_initial_or_identity(self):
        duadd = vectorize('int64(int64, int64)')(pyuadd)
        self._check_reduce(duadd, dtype=np.int64)

    def test_invalid_input(self):
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a):
            return duadd.reduce(a)
        exc_msg = 'The first argument "array" must be array-like'
        with self.assertRaisesRegex(TypingError, exc_msg):
            foo('a')

    def test_dufunc_negative_axis(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a, axis):
            return duadd.reduce(a, axis=axis)
        a = np.arange(40).reshape(5, 4, 2)
        cases = (0, -1, (0, -1), (-1, -2), (1, -1), -3)
        for axis in cases:
            expected = duadd.reduce(a, axis)
            got = foo(a, axis)
            self.assertPreciseEqual(expected, got)

    def test_dufunc_invalid_axis(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a, axis):
            return duadd.reduce(a, axis=axis)
        a = np.arange(40).reshape(5, 4, 2)
        cases = ((0, 0), (0, 1, 0), (0, -3), (-1, -1), (-1, 2))
        for axis in cases:
            msg = "duplicate value in 'axis'"
            with self.assertRaisesRegex(ValueError, msg):
                foo(a, axis)
        cases = (-4, 3, (0, -4))
        for axis in cases:
            with self.assertRaisesRegex(ValueError, 'Invalid axis'):
                foo(a, axis)