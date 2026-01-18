import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestExpandDims:

    def test_functionality(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        for axis in range(-5, 4):
            b = expand_dims(a, axis)
            assert_(b.shape[axis] == 1)
            assert_(np.squeeze(b).shape == s)

    def test_axis_tuple(self):
        a = np.empty((3, 3, 3))
        assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        assert_raises(np.AxisError, expand_dims, a, -6)
        assert_raises(np.AxisError, expand_dims, a, 5)
        a = np.empty((3, 3, 3))
        assert_raises(np.AxisError, expand_dims, a, (0, -6))
        assert_raises(np.AxisError, expand_dims, a, (0, 5))

    def test_repeated_axis(self):
        a = np.empty((3, 3, 3))
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))

    def test_subclasses(self):
        a = np.arange(10).reshape((2, 5))
        a = np.ma.array(a, mask=a % 3 == 0)
        expanded = np.expand_dims(a, axis=1)
        assert_(isinstance(expanded, np.ma.MaskedArray))
        assert_equal(expanded.shape, (2, 1, 5))
        assert_equal(expanded.mask.shape, (2, 1, 5))