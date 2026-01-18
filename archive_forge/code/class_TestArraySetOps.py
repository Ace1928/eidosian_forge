import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
class TestArraySetOps:

    def test_unique_onlist(self):
        data = [1, 1, 1, 2, 2, 3]
        test = unique(data, return_index=True, return_inverse=True)
        assert_(isinstance(test[0], MaskedArray))
        assert_equal(test[0], masked_array([1, 2, 3], mask=[0, 0, 0]))
        assert_equal(test[1], [0, 3, 5])
        assert_equal(test[2], [0, 0, 0, 1, 1, 2])

    def test_unique_onmaskedarray(self):
        data = masked_array([1, 1, 1, 2, 2, 3], mask=[0, 0, 1, 0, 1, 0])
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])
        data.fill_value = 3
        data = masked_array(data=[1, 1, 1, 2, 2, 3], mask=[0, 0, 1, 0, 1, 0], fill_value=3)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])

    def test_unique_allmasked(self):
        data = masked_array([1, 1, 1], mask=True)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1], mask=[True]))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0, 0, 0])
        data = masked
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array(masked))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0])

    def test_ediff1d(self):
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        control = array([1, 1, 1, 4], mask=[1, 0, 0, 1])
        test = ediff1d(x)
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_tobegin(self):
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_begin=masked)
        control = array([0, 1, 1, 1, 4], mask=[1, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        test = ediff1d(x, to_begin=[1, 2, 3])
        control = array([1, 2, 3, 1, 1, 1, 4], mask=[0, 0, 0, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_toend(self):
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_end=masked)
        control = array([1, 1, 1, 4, 0], mask=[1, 0, 0, 1, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        test = ediff1d(x, to_end=[1, 2, 3])
        control = array([1, 1, 1, 4, 1, 2, 3], mask=[1, 0, 0, 1, 0, 0, 0])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_tobegin_toend(self):
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 4, 0], mask=[1, 1, 0, 0, 1, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        test = ediff1d(x, to_end=[1, 2, 3], to_begin=masked)
        control = array([0, 1, 1, 1, 4, 1, 2, 3], mask=[1, 1, 0, 0, 1, 0, 0, 0])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_ndarray(self):
        x = np.arange(5)
        test = ediff1d(x)
        control = array([1, 1, 1, 1], mask=[0, 0, 0, 0])
        assert_equal(test, control)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 1, 0], mask=[1, 0, 0, 0, 0, 1])
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_intersect1d(self):
        x = array([1, 3, 3, 3], mask=[0, 0, 0, 1])
        y = array([3, 1, 1, 1], mask=[0, 0, 0, 1])
        test = intersect1d(x, y)
        control = array([1, 3, -1], mask=[0, 0, 1])
        assert_equal(test, control)

    def test_setxor1d(self):
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7]))
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = [1, 2, 3, 4, 5]
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7, -1], mask=[0, 0, 0, 1]))
        a = array([1, 2, 3])
        b = array([6, 5, 4])
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        a = array([1, 8, 2, 3], mask=[0, 1, 0, 0])
        b = array([6, 5, 4, 8], mask=[0, 0, 0, 1])
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        assert_array_equal([], setxor1d([], []))

    def test_isin(self):
        a = np.arange(24).reshape([2, 3, 4])
        mask = np.zeros([2, 3, 4])
        mask[1, 2, 0] = 1
        a = array(a, mask=mask)
        b = array(data=[0, 10, 20, 30, 1, 3, 11, 22, 33], mask=[0, 1, 0, 1, 0, 1, 0, 1, 0])
        ec = zeros((2, 3, 4), dtype=bool)
        ec[0, 0, 0] = True
        ec[0, 0, 1] = True
        ec[0, 2, 3] = True
        c = isin(a, b)
        assert_(isinstance(c, MaskedArray))
        assert_array_equal(c, ec)
        d = np.isin(a, b[~b.mask]) & ~a.mask
        assert_array_equal(c, d)

    def test_in1d(self):
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = in1d(a, b)
        assert_equal(test, [True, True, True, False, True])
        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        test = in1d(a, b)
        assert_equal(test, [True, True, False, True, True])
        assert_array_equal([], in1d([], []))

    def test_in1d_invert(self):
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))
        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))
        assert_array_equal([], in1d([], [], invert=True))

    def test_union1d(self):
        a = array([1, 2, 5, 7, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = union1d(a, b)
        control = array([1, 2, 3, 4, 5, 7, -1], mask=[0, 0, 0, 0, 0, 0, 1])
        assert_equal(test, control)
        x = array([[0, 1, 2], [3, 4, 5]], mask=[[0, 0, 0], [0, 0, 1]])
        y = array([0, 1, 2, 3, 4], mask=[0, 0, 0, 0, 1])
        ez = array([0, 1, 2, 3, 4, 5], mask=[0, 0, 0, 0, 0, 1])
        z = union1d(x, y)
        assert_equal(z, ez)
        assert_array_equal([], union1d([], []))

    def test_setdiff1d(self):
        a = array([6, 5, 4, 7, 7, 1, 2, 1], mask=[0, 0, 0, 0, 0, 0, 0, 1])
        b = array([2, 4, 3, 3, 2, 1, 5])
        test = setdiff1d(a, b)
        assert_equal(test, array([6, 7, -1], mask=[0, 0, 1]))
        a = arange(10)
        b = arange(8)
        assert_equal(setdiff1d(a, b), array([8, 9]))
        a = array([], np.uint32, mask=[])
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    def test_setdiff1d_char_array(self):
        a = np.array(['a', 'b', 'c'])
        b = np.array(['a', 'b', 's'])
        assert_array_equal(setdiff1d(a, b), np.array(['c']))