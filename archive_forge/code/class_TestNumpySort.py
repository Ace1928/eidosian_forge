import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
class TestNumpySort(TestCase):

    def setUp(self):
        np.random.seed(42)

    def int_arrays(self):
        for size in (5, 20, 50, 500):
            yield np.random.randint(99, size=size)

    def float_arrays(self):
        for size in (5, 20, 50, 500):
            yield (np.random.random(size=size) * 100)
        for size in (5, 20, 50, 500):
            orig = np.random.random(size=size) * 100
            orig[np.random.random(size=size) < 0.1] = float('nan')
            yield orig
        for size in (50, 500):
            orig = np.random.random(size=size) * 100
            orig[np.random.random(size=size) < 0.9] = float('nan')
            yield orig

    def has_duplicates(self, arr):
        """
        Whether the array has duplicates.  Takes NaNs into account.
        """
        if np.count_nonzero(np.isnan(arr)) > 1:
            return True
        if np.unique(arr).size < arr.size:
            return True
        return False

    def check_sort_inplace(self, pyfunc, cfunc, val):
        expected = copy.copy(val)
        got = copy.copy(val)
        pyfunc(expected)
        cfunc(got)
        self.assertPreciseEqual(got, expected)

    def check_sort_copy(self, pyfunc, cfunc, val):
        orig = copy.copy(val)
        expected = pyfunc(val)
        got = cfunc(val)
        self.assertPreciseEqual(got, expected)
        self.assertPreciseEqual(val, orig)

    def check_argsort(self, pyfunc, cfunc, val, kwargs={}):
        orig = copy.copy(val)
        expected = pyfunc(val, **kwargs)
        got = cfunc(val, **kwargs)
        self.assertPreciseEqual(orig[got], np.sort(orig), msg="the array wasn't argsorted")
        if not self.has_duplicates(orig):
            self.assertPreciseEqual(got, expected)
        self.assertPreciseEqual(val, orig)

    def test_array_sort_int(self):
        pyfunc = sort_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for orig in self.int_arrays():
            self.check_sort_inplace(pyfunc, cfunc, orig)

    def test_array_sort_float(self):
        pyfunc = sort_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for orig in self.float_arrays():
            self.check_sort_inplace(pyfunc, cfunc, orig)

    def test_array_sort_complex(self):
        pyfunc = sort_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for real in self.float_arrays():
            imag = real[:]
            np.random.shuffle(imag)
            orig = np.array([complex(*x) for x in zip(real, imag)])
            self.check_sort_inplace(pyfunc, cfunc, orig)

    def test_np_sort_int(self):
        pyfunc = np_sort_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for orig in self.int_arrays():
            self.check_sort_copy(pyfunc, cfunc, orig)

    def test_np_sort_float(self):
        pyfunc = np_sort_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for size in (5, 20, 50, 500):
            orig = np.random.random(size=size) * 100
            orig[np.random.random(size=size) < 0.1] = float('nan')
            self.check_sort_copy(pyfunc, cfunc, orig)

    def test_np_sort_complex(self):
        pyfunc = np_sort_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for size in (5, 20, 50, 500):
            real = np.random.random(size=size) * 100
            imag = np.random.random(size=size) * 100
            real[np.random.random(size=size) < 0.1] = float('nan')
            imag[np.random.random(size=size) < 0.1] = float('nan')
            orig = np.array([complex(*x) for x in zip(real, imag)])
            self.check_sort_copy(pyfunc, cfunc, orig)

    def test_argsort_int(self):

        def check(pyfunc):
            cfunc = jit(nopython=True)(pyfunc)
            for orig in self.int_arrays():
                self.check_argsort(pyfunc, cfunc, orig)
        check(argsort_usecase)
        check(np_argsort_usecase)

    def test_argsort_kind_int(self):

        def check(pyfunc, is_stable):
            cfunc = jit(nopython=True)(pyfunc)
            for orig in self.int_arrays():
                self.check_argsort(pyfunc, cfunc, orig, dict(is_stable=is_stable))
        check(argsort_kind_usecase, is_stable=True)
        check(np_argsort_kind_usecase, is_stable=True)
        check(argsort_kind_usecase, is_stable=False)
        check(np_argsort_kind_usecase, is_stable=False)

    def test_argsort_float(self):

        def check(pyfunc):
            cfunc = jit(nopython=True)(pyfunc)
            for orig in self.float_arrays():
                self.check_argsort(pyfunc, cfunc, orig)
        check(argsort_usecase)
        check(np_argsort_usecase)

    def test_argsort_float_supplemental(self):

        def check(pyfunc, is_stable):
            cfunc = jit(nopython=True)(pyfunc)
            for orig in self.float_arrays():
                self.check_argsort(pyfunc, cfunc, orig, dict(is_stable=is_stable))
        check(argsort_kind_usecase, is_stable=True)
        check(np_argsort_kind_usecase, is_stable=True)
        check(argsort_kind_usecase, is_stable=False)
        check(np_argsort_kind_usecase, is_stable=False)

    def test_argsort_complex(self):

        def check(pyfunc):
            cfunc = jit(nopython=True)(pyfunc)
            for real in self.float_arrays():
                imag = real[:]
                np.random.shuffle(imag)
                orig = np.array([complex(*x) for x in zip(real, imag)])
                self.check_argsort(pyfunc, cfunc, orig)
        check(argsort_usecase)
        check(np_argsort_usecase)

    def test_argsort_complex_supplemental(self):

        def check(pyfunc, is_stable):
            cfunc = jit(nopython=True)(pyfunc)
            for real in self.float_arrays():
                imag = real[:]
                np.random.shuffle(imag)
                orig = np.array([complex(*x) for x in zip(real, imag)])
                self.check_argsort(pyfunc, cfunc, orig, dict(is_stable=is_stable))
        check(argsort_kind_usecase, is_stable=True)
        check(np_argsort_kind_usecase, is_stable=True)
        check(argsort_kind_usecase, is_stable=False)
        check(np_argsort_kind_usecase, is_stable=False)

    def test_bad_array(self):
        cfunc = jit(nopython=True)(np_sort_usecase)
        msg = '.*Argument "a" must be array-like.*'
        with self.assertRaisesRegex(errors.TypingError, msg) as raises:
            cfunc(None)