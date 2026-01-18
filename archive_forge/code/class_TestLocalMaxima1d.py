import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
class TestLocalMaxima1d:

    def test_empty(self):
        """Test with empty signal."""
        x = np.array([], dtype=np.float64)
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    def test_linear(self):
        """Test with linear signal."""
        x = np.linspace(0, 100)
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    def test_simple(self):
        """Test with simple signal."""
        x = np.linspace(-10, 10, 50)
        x[2::3] += 1
        expected = np.arange(2, 50, 3)
        for array in _local_maxima_1d(x):
            assert_equal(array, expected)
            assert_(array.base is None)

    def test_flat_maxima(self):
        """Test if flat maxima are detected correctly."""
        x = np.array([-1.3, 0, 1, 0, 2, 2, 0, 3, 3, 3, 2.99, 4, 4, 4, 4, -10, -5, -5, -5, -5, -5, -10])
        midpoints, left_edges, right_edges = _local_maxima_1d(x)
        assert_equal(midpoints, np.array([2, 4, 8, 12, 18]))
        assert_equal(left_edges, np.array([2, 4, 7, 11, 16]))
        assert_equal(right_edges, np.array([2, 5, 9, 14, 20]))

    @pytest.mark.parametrize('x', [np.array([1.0, 0, 2]), np.array([3.0, 3, 0, 4, 4]), np.array([5.0, 5, 5, 0, 6, 6, 6])])
    def test_signal_edges(self, x):
        """Test if behavior on signal edges is correct."""
        for array in _local_maxima_1d(x):
            assert_equal(array, np.array([]))
            assert_(array.base is None)

    def test_exceptions(self):
        """Test input validation and raised exceptions."""
        with raises(ValueError, match='wrong number of dimensions'):
            _local_maxima_1d(np.ones((1, 1)))
        with raises(ValueError, match="expected 'const float64_t'"):
            _local_maxima_1d(np.ones(1, dtype=int))
        with raises(TypeError, match='list'):
            _local_maxima_1d([1.0, 2.0])
        with raises(TypeError, match="'x' must not be None"):
            _local_maxima_1d(None)