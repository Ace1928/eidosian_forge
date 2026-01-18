import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
class TestLoopTypesDatetime(_LoopTypesTester):
    _ufuncs = supported_ufuncs[:]
    _ufuncs.remove(np.divmod)
    _required_types = 'mM'

    def test_add(self):
        ufunc = np.add
        fn = _make_ufunc_usecase(ufunc)
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', 'm8[ms]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'm8[ms]'])
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'm8[m]'])

    def test_subtract(self):
        ufunc = np.subtract
        fn = _make_ufunc_usecase(ufunc)
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[s]', 'M8[m]', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[s]', 'M8[m]', 'm8[ms]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', 'm8[ms]'])
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', 'm8[m]'])

    def test_multiply(self):
        ufunc = np.multiply
        fn = _make_ufunc_usecase(ufunc)
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[us]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['q', 'm8[s]', 'm8[us]'])
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[m]'])

    def test_true_divide(self):
        ufunc = np.true_divide
        fn = _make_ufunc_usecase(ufunc)
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', 'd'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', 'd'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'q', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'd', 'm8[s]'])
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[m]'])

    def test_floor_divide(self):
        ufunc = np.floor_divide
        fn = _make_ufunc_usecase(ufunc)
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'q', 'm8[s]'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'd', 'm8[s]'])
        with self.assertRaises(LoweringError):
            self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[m]'])

    def _check_comparison(self, ufunc):
        fn = _make_ufunc_usecase(ufunc)
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[m]', 'm8[s]', '?'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'm8[m]', '?'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[m]', 'M8[s]', '?'])
        self._check_ufunc_with_dtypes(fn, ufunc, ['M8[s]', 'M8[m]', '?'])

    def test_comparisons(self):
        for ufunc in [np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal]:
            self._check_comparison(ufunc)