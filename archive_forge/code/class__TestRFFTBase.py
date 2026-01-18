from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class _TestRFFTBase:

    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        for t in [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4, 5]]:
            x = np.array(t, dtype=self.rdt)
            y = rfft(x)
            y1 = direct_rdft(x)
            assert_array_almost_equal(y, y1)
            assert_equal(y.dtype, self.cdt)

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2 ** i
            x = np.arange(n)
            y1 = np.fft.rfft(x)
            y = rfft(x)
            assert_array_almost_equal(y, y1)

    def test_invalid_sizes(self):
        assert_raises(ValueError, rfft, [])
        assert_raises(ValueError, rfft, [[1, 1], [2, 2]], -5)

    def test_complex_input(self):
        x = np.zeros(10, dtype=self.cdt)
        with assert_raises(TypeError, match='x must be a real sequence'):
            rfft(x)

    class MockSeries:

        def __init__(self, data):
            self.data = np.asarray(data)

        def __getattr__(self, item):
            try:
                return getattr(self.data, item)
            except AttributeError as e:
                raise AttributeError(f"'MockSeries' object has no attribute '{item}'") from e

    def test_non_ndarray_with_dtype(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        xs = _TestRFFTBase.MockSeries(x)
        expected = [1, 2, 3, 4, 5]
        rfft(xs)
        assert_equal(x, expected)
        assert_equal(xs.data, expected)