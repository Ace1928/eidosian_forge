from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class _TestFFTBase:

    def setup_method(self):
        self.cdt = None
        self.rdt = None
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([1, 2, 3, 4 + 1j, 1, 2, 3, 4 + 2j], dtype=self.cdt)
        y = fft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_dft(x)
        assert_array_almost_equal(y, y1)
        x = np.array([1, 2, 3, 4 + 0j, 5], dtype=self.cdt)
        assert_array_almost_equal(fft(x), direct_dft(x))

    def test_n_argument_real(self):
        x1 = np.array([1, 2, 3, 4], dtype=self.rdt)
        x2 = np.array([1, 2, 3, 4], dtype=self.rdt)
        y = fft([x1, x2], n=4)
        assert_equal(y.dtype, self.cdt)
        assert_equal(y.shape, (2, 4))
        assert_array_almost_equal(y[0], direct_dft(x1))
        assert_array_almost_equal(y[1], direct_dft(x2))

    def _test_n_argument_complex(self):
        x1 = np.array([1, 2, 3, 4 + 1j], dtype=self.cdt)
        x2 = np.array([1, 2, 3, 4 + 1j], dtype=self.cdt)
        y = fft([x1, x2], n=4)
        assert_equal(y.dtype, self.cdt)
        assert_equal(y.shape, (2, 4))
        assert_array_almost_equal(y[0], direct_dft(x1))
        assert_array_almost_equal(y[1], direct_dft(x2))

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2 ** i
            x = np.arange(n)
            y = fft(x.astype(complex))
            y2 = numpy.fft.fft(x)
            assert_array_almost_equal(y, y2)
            y = fft(x)
            assert_array_almost_equal(y, y2)

    def test_invalid_sizes(self):
        assert_raises(ValueError, fft, [])
        assert_raises(ValueError, fft, [[1, 1], [2, 2]], -5)