import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestArrayEqual(_GenericTest):

    def setup_method(self):
        self._assert_func = assert_array_equal

    def test_generic_rank1(self):
        """Test rank 1 array for all dtypes."""

        def foo(t):
            a = np.empty(2, t)
            a.fill(1)
            b = a.copy()
            c = a.copy()
            c.fill(0)
            self._test_equal(a, b)
            self._test_not_equal(c, b)
        for t in '?bhilqpBHILQPfdgFDG':
            foo(t)
        for t in ['S1', 'U1']:
            foo(t)

    def test_0_ndim_array(self):
        x = np.array(473963742225900817127911193656584771)
        y = np.array(18535119325151578301457182298393896)
        assert_raises(AssertionError, self._assert_func, x, y)
        y = x
        self._assert_func(x, y)
        x = np.array(43)
        y = np.array(10)
        assert_raises(AssertionError, self._assert_func, x, y)
        y = x
        self._assert_func(x, y)

    def test_generic_rank3(self):
        """Test rank 3 array for all dtypes."""

        def foo(t):
            a = np.empty((4, 2, 3), t)
            a.fill(1)
            b = a.copy()
            c = a.copy()
            c.fill(0)
            self._test_equal(a, b)
            self._test_not_equal(c, b)
        for t in '?bhilqpBHILQPfdgFDG':
            foo(t)
        for t in ['S1', 'U1']:
            foo(t)

    def test_nan_array(self):
        """Test arrays with nan values in them."""
        a = np.array([1, 2, np.nan])
        b = np.array([1, 2, np.nan])
        self._test_equal(a, b)
        c = np.array([1, 2, 3])
        self._test_not_equal(c, b)

    def test_string_arrays(self):
        """Test two arrays with different shapes are found not equal."""
        a = np.array(['floupi', 'floupa'])
        b = np.array(['floupi', 'floupa'])
        self._test_equal(a, b)
        c = np.array(['floupipi', 'floupa'])
        self._test_not_equal(c, b)

    def test_recarrays(self):
        """Test record arrays."""
        a = np.empty(2, [('floupi', float), ('floupa', float)])
        a['floupi'] = [1, 2]
        a['floupa'] = [1, 2]
        b = a.copy()
        self._test_equal(a, b)
        c = np.empty(2, [('floupipi', float), ('floupi', float), ('floupa', float)])
        c['floupipi'] = a['floupi'].copy()
        c['floupa'] = a['floupa'].copy()
        with pytest.raises(TypeError):
            self._test_not_equal(c, b)

    def test_masked_nan_inf(self):
        a = np.ma.MaskedArray([3.0, 4.0, 6.5], mask=[False, True, False])
        b = np.array([3.0, np.nan, 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.MaskedArray([3.0, 4.0, 6.5], mask=[True, False, False])
        b = np.array([np.inf, 4.0, 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)

    def test_subclass_that_overrides_eq(self):

        class MyArray(np.ndarray):

            def __eq__(self, other):
                return bool(np.equal(self, other).all())

            def __ne__(self, other):
                return not self == other
        a = np.array([1.0, 2.0]).view(MyArray)
        b = np.array([2.0, 3.0]).view(MyArray)
        assert_(type(a == a), bool)
        assert_(a == a)
        assert_(a != b)
        self._test_equal(a, a)
        self._test_not_equal(a, b)
        self._test_not_equal(b, a)

    def test_subclass_that_does_not_implement_npall(self):

        class MyArray(np.ndarray):

            def __array_function__(self, *args, **kwargs):
                return NotImplemented
        a = np.array([1.0, 2.0]).view(MyArray)
        b = np.array([2.0, 3.0]).view(MyArray)
        with assert_raises(TypeError):
            np.all(a)
        self._test_equal(a, a)
        self._test_not_equal(a, b)
        self._test_not_equal(b, a)

    def test_suppress_overflow_warnings(self):
        with pytest.raises(AssertionError):
            with np.errstate(all='raise'):
                np.testing.assert_array_equal(np.array([1, 2, 3], np.float32), np.array([1, 1e-40, 3], np.float32))

    def test_array_vs_scalar_is_equal(self):
        """Test comparing an array with a scalar when all values are equal."""
        a = np.array([1.0, 1.0, 1.0])
        b = 1.0
        self._test_equal(a, b)

    def test_array_vs_scalar_not_equal(self):
        """Test comparing an array with a scalar when not all values equal."""
        a = np.array([1.0, 2.0, 3.0])
        b = 1.0
        self._test_not_equal(a, b)

    def test_array_vs_scalar_strict(self):
        """Test comparing an array with a scalar with strict option."""
        a = np.array([1.0, 1.0, 1.0])
        b = 1.0
        with pytest.raises(AssertionError):
            assert_array_equal(a, b, strict=True)

    def test_array_vs_array_strict(self):
        """Test comparing two arrays with strict option."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        assert_array_equal(a, b, strict=True)

    def test_array_vs_float_array_strict(self):
        """Test comparing two arrays with strict option."""
        a = np.array([1, 1, 1])
        b = np.array([1.0, 1.0, 1.0])
        with pytest.raises(AssertionError):
            assert_array_equal(a, b, strict=True)