import sys
import numpy as np
from numpy.testing import (assert_, assert_approx_equal,
import pytest
from pytest import raises as assert_raises
from scipy import signal
from scipy.fft import fftfreq
from scipy.integrate import trapezoid
from scipy.signal import (periodogram, welch, lombscargle, coherence,
from scipy.signal._spectral_py import _spectral_helper
from scipy.signal.tests._scipy_spectral_test_shim import stft_compare as stft
from scipy.signal.tests._scipy_spectral_test_shim import istft_compare as istft
from scipy.signal.tests._scipy_spectral_test_shim import csd_compare as csd
class TestPeriodogram:

    def test_real_onesided_even(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_real_onesided_odd(self):
        x = np.zeros(15)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.arange(8.0) / 15.0)
        q = np.ones(8)
        q[0] = 0
        q *= 2.0 / 15.0
        assert_allclose(p, q, atol=1e-15)

    def test_real_twosided(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 1 / 16.0)
        q[0] = 0
        assert_allclose(p, q)

    def test_real_spectrum(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, scaling='spectrum')
        g, q = periodogram(x, scaling='density')
        assert_allclose(f, np.linspace(0, 0.5, 9))
        assert_allclose(p, q / 16.0)

    def test_integer_even(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_integer_odd(self):
        x = np.zeros(15, dtype=int)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.arange(8.0) / 15.0)
        q = np.ones(8)
        q[0] = 0
        q *= 2.0 / 15.0
        assert_allclose(p, q, atol=1e-15)

    def test_integer_twosided(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 1 / 16.0)
        q[0] = 0
        assert_allclose(p, q)

    def test_complex(self):
        x = np.zeros(16, np.complex128)
        x[0] = 1.0 + 2j
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 5.0 / 16.0)
        q[0] = 0
        assert_allclose(p, q)

    def test_unk_scaling(self):
        assert_raises(ValueError, periodogram, np.zeros(4, np.complex128), scaling='foo')

    @pytest.mark.skipif(sys.maxsize <= 2 ** 32, reason='On some 32-bit tolerance issue')
    def test_nd_axis_m1(self):
        x = np.zeros(20, dtype=np.float64)
        x = x.reshape((2, 1, 10))
        x[:, :, 0] = 1.0
        f, p = periodogram(x)
        assert_array_equal(p.shape, (2, 1, 6))
        assert_array_almost_equal_nulp(p[0, 0, :], p[1, 0, :], 60)
        f0, p0 = periodogram(x[0, 0, :])
        assert_array_almost_equal_nulp(p0[np.newaxis, :], p[1, :], 60)

    @pytest.mark.skipif(sys.maxsize <= 2 ** 32, reason='On some 32-bit tolerance issue')
    def test_nd_axis_0(self):
        x = np.zeros(20, dtype=np.float64)
        x = x.reshape((10, 2, 1))
        x[0, :, :] = 1.0
        f, p = periodogram(x, axis=0)
        assert_array_equal(p.shape, (6, 2, 1))
        assert_array_almost_equal_nulp(p[:, 0, 0], p[:, 1, 0], 60)
        f0, p0 = periodogram(x[:, 0, 0])
        assert_array_almost_equal_nulp(p0, p[:, 1, 0])

    def test_window_external(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, 10, 'hann')
        win = signal.get_window('hann', 16)
        fe, pe = periodogram(x, 10, win)
        assert_array_almost_equal_nulp(p, pe)
        assert_array_almost_equal_nulp(f, fe)
        win_err = signal.get_window('hann', 32)
        assert_raises(ValueError, periodogram, x, 10, win_err)

    def test_padded_fft(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x)
        fp, pp = periodogram(x, nfft=32)
        assert_allclose(f, fp[::2])
        assert_allclose(p, pp[::2])
        assert_array_equal(pp.shape, (17,))

    def test_empty_input(self):
        f, p = periodogram([])
        assert_array_equal(f.shape, (0,))
        assert_array_equal(p.shape, (0,))
        for shape in [(0,), (3, 0), (0, 5, 2)]:
            f, p = periodogram(np.empty(shape))
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

    def test_empty_input_other_axis(self):
        for shape in [(3, 0), (0, 5, 2)]:
            f, p = periodogram(np.empty(shape), axis=1)
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

    def test_short_nfft(self):
        x = np.zeros(18)
        x[0] = 1
        f, p = periodogram(x, nfft=16)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_nfft_is_xshape(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, nfft=16)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_real_onesided_even_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9, 'f')
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    def test_real_onesided_odd_32(self):
        x = np.zeros(15, 'f')
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.arange(8.0) / 15.0)
        q = np.ones(8, 'f')
        q[0] = 0
        q *= 2.0 / 15.0
        assert_allclose(p, q, atol=1e-07)
        assert_(p.dtype == q.dtype)

    def test_real_twosided_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 1 / 16.0, 'f')
        q[0] = 0
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    def test_complex_32(self):
        x = np.zeros(16, 'F')
        x[0] = 1.0 + 2j
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 5.0 / 16.0, 'f')
        q[0] = 0
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    def test_shorter_window_error(self):
        x = np.zeros(16)
        x[0] = 1
        win = signal.get_window('hann', 10)
        expected_msg = 'the size of the window must be the same size of the input on the specified axis'
        with assert_raises(ValueError, match=expected_msg):
            periodogram(x, window=win)