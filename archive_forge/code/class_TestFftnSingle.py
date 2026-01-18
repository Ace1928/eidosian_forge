from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class TestFftnSingle:

    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y = fftn(np.array(x, np.float32))
        assert_(y.dtype == np.complex64, msg='double precision output with single precision')
        y_r = np.array(fftn(x), np.complex64)
        assert_array_almost_equal_nulp(y, y_r)

    @pytest.mark.parametrize('size', SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    def test_size_accuracy_small(self, size):
        x = np.random.rand(size, size) + 1j * np.random.rand(size, size)
        y1 = fftn(x.real.astype(np.float32))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)
        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2000)

    @pytest.mark.parametrize('size', LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    def test_size_accuracy_large(self, size):
        x = np.random.rand(size, 3) + 1j * np.random.rand(size, 3)
        y1 = fftn(x.real.astype(np.float32))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)
        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2000)

    def test_definition_float16(self):
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y = fftn(np.array(x, np.float16))
        assert_equal(y.dtype, np.complex64)
        y_r = np.array(fftn(x), np.complex64)
        assert_array_almost_equal_nulp(y, y_r)

    @pytest.mark.parametrize('size', SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    def test_float16_input_small(self, size):
        x = np.random.rand(size, size) + 1j * np.random.rand(size, size)
        y1 = fftn(x.real.astype(np.float16))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)
        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 500000.0)

    @pytest.mark.parametrize('size', LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    def test_float16_input_large(self, size):
        x = np.random.rand(size, 3) + 1j * np.random.rand(size, 3)
        y1 = fftn(x.real.astype(np.float16))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)
        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2000000.0)