import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
class TestLinearRamp:

    def test_check_simple(self):
        a = np.arange(100).astype('f')
        a = np.pad(a, (25, 20), 'linear_ramp', end_values=(4, 5))
        b = np.array([4.0, 3.84, 3.68, 3.52, 3.36, 3.2, 3.04, 2.88, 2.72, 2.56, 2.4, 2.24, 2.08, 1.92, 1.76, 1.6, 1.44, 1.28, 1.12, 0.96, 0.8, 0.64, 0.48, 0.32, 0.16, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 94.3, 89.6, 84.9, 80.2, 75.5, 70.8, 66.1, 61.4, 56.7, 52.0, 47.3, 42.6, 37.9, 33.2, 28.5, 23.8, 19.1, 14.4, 9.7, 5.0])
        assert_allclose(a, b, rtol=1e-05, atol=1e-05)

    def test_check_2d(self):
        arr = np.arange(20).reshape(4, 5).astype(np.float64)
        test = np.pad(arr, (2, 2), mode='linear_ramp', end_values=(0, 0))
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 0.0], [0.0, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0, 4.5, 0.0], [0.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 7.0, 0.0], [0.0, 7.5, 15.0, 16.0, 17.0, 18.0, 19.0, 9.5, 0.0], [0.0, 3.75, 7.5, 8.0, 8.5, 9.0, 9.5, 4.75, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert_allclose(test, expected)

    @pytest.mark.xfail(exceptions=(AssertionError,))
    def test_object_array(self):
        from fractions import Fraction
        arr = np.array([Fraction(1, 2), Fraction(-1, 2)])
        actual = np.pad(arr, (2, 3), mode='linear_ramp', end_values=0)
        expected = np.array([Fraction(0, 12), Fraction(3, 12), Fraction(6, 12), Fraction(-6, 12), Fraction(-4, 12), Fraction(-2, 12), Fraction(-0, 12)])
        assert_equal(actual, expected)

    def test_end_values(self):
        """Ensure that end values are exact."""
        a = np.pad(np.ones(10).reshape(2, 5), (223, 123), mode='linear_ramp')
        assert_equal(a[:, 0], 0.0)
        assert_equal(a[:, -1], 0.0)
        assert_equal(a[0, :], 0.0)
        assert_equal(a[-1, :], 0.0)

    @pytest.mark.parametrize('dtype', _numeric_dtypes)
    def test_negative_difference(self, dtype):
        """
        Check correct behavior of unsigned dtypes if there is a negative
        difference between the edge to pad and `end_values`. Check both cases
        to be independent of implementation. Test behavior for all other dtypes
        in case dtype casting interferes with complex dtypes. See gh-14191.
        """
        x = np.array([3], dtype=dtype)
        result = np.pad(x, 3, mode='linear_ramp', end_values=0)
        expected = np.array([0, 1, 2, 3, 2, 1, 0], dtype=dtype)
        assert_equal(result, expected)
        x = np.array([0], dtype=dtype)
        result = np.pad(x, 3, mode='linear_ramp', end_values=3)
        expected = np.array([3, 2, 1, 0, 1, 2, 3], dtype=dtype)
        assert_equal(result, expected)