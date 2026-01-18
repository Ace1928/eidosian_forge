import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
class TestSlidingWindowView:

    def test_1d(self):
        arr = np.arange(5)
        arr_view = sliding_window_view(arr, 2)
        expected = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        assert_array_equal(arr_view, expected)

    def test_2d(self):
        i, j = np.ogrid[:3, :4]
        arr = 10 * i + j
        shape = (2, 2)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1], [10, 11]], [[1, 2], [11, 12]], [[2, 3], [12, 13]]], [[[10, 11], [20, 21]], [[11, 12], [21, 22]], [[12, 13], [22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_with_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10 * i + j
        arr_view = sliding_window_view(arr, 3, 0)
        expected = np.array([[[0, 10, 20], [1, 11, 21], [2, 12, 22], [3, 13, 23]]])
        assert_array_equal(arr_view, expected)

    def test_2d_repeated_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10 * i + j
        arr_view = sliding_window_view(arr, (2, 3), (1, 1))
        expected = np.array([[[[0, 1, 2], [1, 2, 3]]], [[[10, 11, 12], [11, 12, 13]]], [[[20, 21, 22], [21, 22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_without_axis(self):
        i, j = np.ogrid[:4, :4]
        arr = 10 * i + j
        shape = (2, 3)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1, 2], [10, 11, 12]], [[1, 2, 3], [11, 12, 13]]], [[[10, 11, 12], [20, 21, 22]], [[11, 12, 13], [21, 22, 23]]], [[[20, 21, 22], [30, 31, 32]], [[21, 22, 23], [31, 32, 33]]]])
        assert_array_equal(arr_view, expected)

    def test_errors(self):
        i, j = np.ogrid[:4, :4]
        arr = 10 * i + j
        with pytest.raises(ValueError, match='cannot contain negative values'):
            sliding_window_view(arr, (-1, 3))
        with pytest.raises(ValueError, match='must provide window_shape for all dimensions of `x`'):
            sliding_window_view(arr, (1,))
        with pytest.raises(ValueError, match='Must provide matching length window_shape and axis'):
            sliding_window_view(arr, (1, 3, 4), axis=(0, 1))
        with pytest.raises(ValueError, match='window shape cannot be larger than input array'):
            sliding_window_view(arr, (5, 5))

    def test_writeable(self):
        arr = np.arange(5)
        view = sliding_window_view(arr, 2, writeable=False)
        assert_(not view.flags.writeable)
        with pytest.raises(ValueError, match='assignment destination is read-only'):
            view[0, 0] = 3
        view = sliding_window_view(arr, 2, writeable=True)
        assert_(view.flags.writeable)
        view[0, 1] = 3
        assert_array_equal(arr, np.array([0, 3, 2, 3, 4]))

    def test_subok(self):

        class MyArray(np.ndarray):
            pass
        arr = np.arange(5).view(MyArray)
        assert_(not isinstance(sliding_window_view(arr, 2, subok=False), MyArray))
        assert_(isinstance(sliding_window_view(arr, 2, subok=True), MyArray))
        assert_(not isinstance(sliding_window_view(arr, 2), MyArray))