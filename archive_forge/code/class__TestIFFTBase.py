from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class _TestIFFTBase:

    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([1, 2, 3, 4 + 1j, 1, 2, 3, 4 + 2j], self.cdt)
        y = ifft(x)
        y1 = direct_idft(x)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(y, y1)
        x = np.array([1, 2, 3, 4 + 0j, 5], self.cdt)
        assert_array_almost_equal(ifft(x), direct_idft(x))

    def test_definition_real(self):
        x = np.array([1, 2, 3, 4, 1, 2, 3, 4], self.rdt)
        y = ifft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_idft(x)
        assert_array_almost_equal(y, y1)
        x = np.array([1, 2, 3, 4, 5], dtype=self.rdt)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(ifft(x), direct_idft(x))

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2 ** i
            x = np.arange(n)
            y = ifft(x.astype(self.cdt))
            y2 = numpy.fft.ifft(x)
            assert_allclose(y, y2, rtol=self.rtol, atol=self.atol)
            y = ifft(x)
            assert_allclose(y, y2, rtol=self.rtol, atol=self.atol)

    def test_random_complex(self):
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            x = random([size]).astype(self.cdt)
            x = random([size]).astype(self.cdt) + 1j * x
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)

    def test_random_real(self):
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            x = random([size]).astype(self.rdt)
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)

    def test_size_accuracy(self):
        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            np.random.seed(1234)
            x = np.random.rand(size).astype(self.rdt)
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            x = (x + 1j * np.random.rand(size)).astype(self.cdt)
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)

    def test_invalid_sizes(self):
        assert_raises(ValueError, ifft, [])
        assert_raises(ValueError, ifft, [[1, 1], [2, 2]], -5)