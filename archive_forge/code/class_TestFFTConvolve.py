import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
class TestFFTConvolve:

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_real(self, axes):
        a = array([1, 2, 3])
        expected = array([1, 4, 10, 12, 9.0])
        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_real_axes(self, axes):
        a = array([1, 2, 3])
        expected = array([1, 4, 10, 12, 9.0])
        a = np.tile(a, [2, 1])
        expected = np.tile(expected, [2, 1])
        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_complex(self, axes):
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])
        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_complex_axes(self, axes):
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])
        a = np.tile(a, [2, 1])
        expected = np.tile(expected, [2, 1])
        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, [0, 1], [1, 0], [0, -1], [-1, 0], [-2, 1], [1, -2], [-2, -1], [-1, -2]])
    def test_2d_real_same(self, axes):
        a = array([[1, 2, 3], [4, 5, 6]])
        expected = array([[1, 4, 10, 12, 9], [8, 26, 56, 54, 36], [16, 40, 73, 60, 36]])
        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[1, 2], [2, 1], [1, -1], [-1, 1], [-2, 2], [2, -2], [-2, -1], [-1, -2]])
    def test_2d_real_same_axes(self, axes):
        a = array([[1, 2, 3], [4, 5, 6]])
        expected = array([[1, 4, 10, 12, 9], [8, 26, 56, 54, 36], [16, 40, 73, 60, 36]])
        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])
        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, [0, 1], [1, 0], [0, -1], [-1, 0], [-2, 1], [1, -2], [-2, -1], [-1, -2]])
    def test_2d_complex_same(self, axes):
        a = array([[1 + 2j, 3 + 4j, 5 + 6j], [2 + 1j, 4 + 3j, 6 + 5j]])
        expected = array([[-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j], [10j, 44j, 118j, 156j, 122j], [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]])
        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[1, 2], [2, 1], [1, -1], [-1, 1], [-2, 2], [2, -2], [-2, -1], [-1, -2]])
    def test_2d_complex_same_axes(self, axes):
        a = array([[1 + 2j, 3 + 4j, 5 + 6j], [2 + 1j, 4 + 3j, 6 + 5j]])
        expected = array([[-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j], [10j, 44j, 118j, 156j, 122j], [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]])
        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])
        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_real_same_mode(self, axes):
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = array([35.0, 41.0, 47.0])
        expected_2 = array([9.0, 20.0, 25.0, 35.0, 41.0, 47.0, 39.0, 28.0, 2.0])
        if axes == '':
            out = fftconvolve(a, b, 'same')
        else:
            out = fftconvolve(a, b, 'same', axes=axes)
        assert_array_almost_equal(out, expected_1)
        if axes == '':
            out = fftconvolve(b, a, 'same')
        else:
            out = fftconvolve(b, a, 'same', axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize('axes', [1, -1, [1], [-1]])
    def test_real_same_mode_axes(self, axes):
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = array([35.0, 41.0, 47.0])
        expected_2 = array([9.0, 20.0, 25.0, 35.0, 41.0, 47.0, 39.0, 28.0, 2.0])
        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected_1 = np.tile(expected_1, [2, 1])
        expected_2 = np.tile(expected_2, [2, 1])
        out = fftconvolve(a, b, 'same', axes=axes)
        assert_array_almost_equal(out, expected_1)
        out = fftconvolve(b, a, 'same', axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_real(self, axes):
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24.0, 31.0, 41.0, 43.0, 49.0, 25.0, 12.0])
        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)
        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1]])
    def test_valid_mode_real_axes(self, axes):
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24.0, 31.0, 41.0, 43.0, 49.0, 25.0, 12.0])
        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])
        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_complex(self, axes):
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = array([45.0 + 12j, 30.0 + 23j, 48 + 32j])
        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)
        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_valid_mode_complex_axes(self, axes):
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = array([45.0 + 12j, 30.0 + 23j, 48 + 32j])
        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])
        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)
        out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    def test_valid_mode_ignore_nonaxes(self):
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24.0, 31.0, 41.0, 43.0, 49.0, 25.0, 12.0])
        a = np.tile(a, [2, 1])
        b = np.tile(b, [1, 1])
        expected = np.tile(expected, [2, 1])
        out = fftconvolve(a, b, 'valid', axes=1)
        assert_array_almost_equal(out, expected)

    def test_empty(self):
        assert_(fftconvolve([], []).size == 0)
        assert_(fftconvolve([5, 6], []).size == 0)
        assert_(fftconvolve([], [7]).size == 0)

    def test_zero_rank(self):
        a = array(4967)
        b = array(3920)
        out = fftconvolve(a, b)
        assert_equal(out, a * b)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        out = fftconvolve(a, b)
        assert_equal(out, a * b)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_random_data(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, 'full')
        if axes == '':
            out = fftconvolve(a, b, 'full')
        else:
            out = fftconvolve(a, b, 'full', axes=axes)
        assert_(np.allclose(out, expected, rtol=1e-10))

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_random_data_axes(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, 'full')
        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])
        out = fftconvolve(a, b, 'full', axes=axes)
        assert_(np.allclose(out, expected, rtol=1e-10))

    @pytest.mark.parametrize('axes', [[1, 4], [4, 1], [1, -1], [-1, 1], [-4, 4], [4, -4], [-4, -1], [-1, -4]])
    def test_random_data_multidim_axes(self, axes):
        a_shape, b_shape = ((123, 22), (132, 11))
        np.random.seed(1234)
        a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
        b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
        expected = convolve2d(a, b, 'full')
        a = a[:, :, None, None, None]
        b = b[:, :, None, None, None]
        expected = expected[:, :, None, None, None]
        a = np.moveaxis(a.swapaxes(0, 2), 1, 4)
        b = np.moveaxis(b.swapaxes(0, 2), 1, 4)
        expected = np.moveaxis(expected.swapaxes(0, 2), 1, 4)
        a = np.tile(a, [2, 1, 3, 1, 1])
        b = np.tile(b, [2, 1, 1, 4, 1])
        expected = np.tile(expected, [2, 1, 3, 4, 1])
        out = fftconvolve(a, b, 'full', axes=axes)
        assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.slow
    @pytest.mark.parametrize('n', list(range(1, 100)) + list(range(1000, 1500)) + np.random.RandomState(1234).randint(1001, 10000, 5).tolist())
    def test_many_sizes(self, n):
        a = np.random.rand(n) + 1j * np.random.rand(n)
        b = np.random.rand(n) + 1j * np.random.rand(n)
        expected = np.convolve(a, b, 'full')
        out = fftconvolve(a, b, 'full')
        assert_allclose(out, expected, atol=1e-10)
        out = fftconvolve(a, b, 'full', axes=[0])
        assert_allclose(out, expected, atol=1e-10)

    def test_fft_nan(self):
        n = 1000
        rng = np.random.default_rng(43876432987)
        sig_nan = rng.standard_normal(n)
        for val in [np.nan, np.inf]:
            sig_nan[100] = val
            coeffs = signal.firwin(200, 0.2)
            msg = 'Use of fft convolution.*|invalid value encountered.*'
            with pytest.warns(RuntimeWarning, match=msg):
                signal.convolve(sig_nan, coeffs, mode='same', method='fft')