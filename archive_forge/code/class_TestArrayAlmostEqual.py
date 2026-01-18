import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestArrayAlmostEqual(_GenericTest):

    def setup_method(self):
        self._assert_func = assert_array_almost_equal

    def test_closeness(self):
        self._assert_func(1.499999, 0.0, decimal=0)
        assert_raises(AssertionError, lambda: self._assert_func(1.5, 0.0, decimal=0))
        self._assert_func([1.499999], [0.0], decimal=0)
        assert_raises(AssertionError, lambda: self._assert_func([1.5], [0.0], decimal=0))

    def test_simple(self):
        x = np.array([1234.2222])
        y = np.array([1234.2223])
        self._assert_func(x, y, decimal=3)
        self._assert_func(x, y, decimal=4)
        assert_raises(AssertionError, lambda: self._assert_func(x, y, decimal=5))

    def test_nan(self):
        anan = np.array([np.nan])
        aone = np.array([1])
        ainf = np.array([np.inf])
        self._assert_func(anan, anan)
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))

    def test_inf(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = a.copy()
        a[0, 0] = np.inf
        assert_raises(AssertionError, lambda: self._assert_func(a, b))
        b[0, 0] = -np.inf
        assert_raises(AssertionError, lambda: self._assert_func(a, b))

    def test_subclass(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.ma.masked_array([[1.0, 2.0], [0.0, 4.0]], [[False, False], [True, False]])
        self._assert_func(a, b)
        self._assert_func(b, a)
        self._assert_func(b, b)
        a = np.ma.MaskedArray(3.5, mask=True)
        b = np.array([3.0, 4.0, 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.masked
        b = np.array([3.0, 4.0, 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.MaskedArray([3.0, 4.0, 6.5], mask=[True, True, True])
        b = np.array([1.0, 2.0, 3.0])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.MaskedArray([3.0, 4.0, 6.5], mask=[True, True, True])
        b = np.array(1.0)
        self._test_equal(a, b)
        self._test_equal(b, a)

    def test_subclass_that_cannot_be_bool(self):

        class MyArray(np.ndarray):

            def __eq__(self, other):
                return super().__eq__(other).view(np.ndarray)

            def __lt__(self, other):
                return super().__lt__(other).view(np.ndarray)

            def all(self, *args, **kwargs):
                raise NotImplementedError
        a = np.array([1.0, 2.0]).view(MyArray)
        self._assert_func(a, a)