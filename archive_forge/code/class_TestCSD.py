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
class TestCSD:

    def test_pad_shorter_x(self):
        x = np.zeros(8)
        y = np.zeros(12)
        f = np.linspace(0, 0.5, 7)
        c = np.zeros(7, dtype=np.complex128)
        f1, c1 = csd(x, y, nperseg=12)
        assert_allclose(f, f1)
        assert_allclose(c, c1)

    def test_pad_shorter_y(self):
        x = np.zeros(12)
        y = np.zeros(8)
        f = np.linspace(0, 0.5, 7)
        c = np.zeros(7, dtype=np.complex128)
        f1, c1 = csd(x, y, nperseg=12)
        assert_allclose(f, f1)
        assert_allclose(c, c1)

    def test_real_onesided_even(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222, 0.11111111])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_real_onesided_odd(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=9)
        assert_allclose(f, np.arange(5.0) / 9.0)
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113, 0.17072113])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_real_twosided(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.07638889])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_real_spectrum(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, scaling='spectrum')
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.015625, 0.02864583, 0.04166667, 0.04166667, 0.02083333])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_integer_onesided_even(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222, 0.11111111])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_integer_onesided_odd(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=9)
        assert_allclose(f, np.arange(5.0) / 9.0)
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113, 0.17072113])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_integer_twosided(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.07638889])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_complex(self):
        x = np.zeros(16, np.complex128)
        x[0] = 1.0 + 2j
        x[8] = 1.0 + 2j
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.41666667, 0.38194444, 0.55555556, 0.55555556, 0.55555556, 0.55555556, 0.55555556, 0.38194444])
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)

    def test_unk_scaling(self):
        assert_raises(ValueError, csd, np.zeros(4, np.complex128), np.ones(4, np.complex128), scaling='foo', nperseg=4)

    def test_detrend_linear(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = csd(x, x, nperseg=10, detrend='linear')
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_no_detrending(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f1, p1 = csd(x, x, nperseg=10, detrend=False)
        f2, p2 = csd(x, x, nperseg=10, detrend=lambda x: x)
        assert_allclose(f1, f2, atol=1e-15)
        assert_allclose(p1, p2, atol=1e-15)

    def test_detrend_external(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = csd(x, x, nperseg=10, detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_detrend_external_nd_m1(self):
        x = np.arange(40, dtype=np.float64) + 0.04
        x = x.reshape((2, 2, 10))
        f, p = csd(x, x, nperseg=10, detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_detrend_external_nd_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2, 1, 10))
        x = np.moveaxis(x, 2, 0)
        f, p = csd(x, x, nperseg=10, axis=0, detrend=lambda seg: signal.detrend(seg, axis=0, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_nd_axis_m1(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2, 1, 10))
        f, p = csd(x, x, nperseg=10)
        assert_array_equal(p.shape, (2, 1, 6))
        assert_allclose(p[0, 0, :], p[1, 0, :], atol=1e-13, rtol=1e-13)
        f0, p0 = csd(x[0, 0, :], x[0, 0, :], nperseg=10)
        assert_allclose(p0[np.newaxis, :], p[1, :], atol=1e-13, rtol=1e-13)

    def test_nd_axis_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((10, 2, 1))
        f, p = csd(x, x, nperseg=10, axis=0)
        assert_array_equal(p.shape, (6, 2, 1))
        assert_allclose(p[:, 0, 0], p[:, 1, 0], atol=1e-13, rtol=1e-13)
        f0, p0 = csd(x[:, 0, 0], x[:, 0, 0], nperseg=10)
        assert_allclose(p0, p[:, 1, 0], atol=1e-13, rtol=1e-13)

    def test_window_external(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, 10, 'hann', 8)
        win = signal.get_window('hann', 8)
        fe, pe = csd(x, x, 10, win, nperseg=None)
        assert_array_almost_equal_nulp(p, pe)
        assert_array_almost_equal_nulp(f, fe)
        assert_array_equal(fe.shape, (5,))
        assert_array_equal(pe.shape, (5,))
        assert_raises(ValueError, csd, x, x, 10, win, nperseg=256)
        win_err = signal.get_window('hann', 32)
        assert_raises(ValueError, csd, x, x, 10, win_err, nperseg=None)

    def test_empty_input(self):
        f, p = csd([], np.zeros(10))
        assert_array_equal(f.shape, (0,))
        assert_array_equal(p.shape, (0,))
        f, p = csd(np.zeros(10), [])
        assert_array_equal(f.shape, (0,))
        assert_array_equal(p.shape, (0,))
        for shape in [(0,), (3, 0), (0, 5, 2)]:
            f, p = csd(np.empty(shape), np.empty(shape))
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)
        f, p = csd(np.ones(10), np.empty((5, 0)))
        assert_array_equal(f.shape, (5, 0))
        assert_array_equal(p.shape, (5, 0))
        f, p = csd(np.empty((5, 0)), np.ones(10))
        assert_array_equal(f.shape, (5, 0))
        assert_array_equal(p.shape, (5, 0))

    def test_empty_input_other_axis(self):
        for shape in [(3, 0), (0, 5, 2)]:
            f, p = csd(np.empty(shape), np.empty(shape), axis=1)
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)
        f, p = csd(np.empty((10, 10, 3)), np.zeros((10, 0, 1)), axis=1)
        assert_array_equal(f.shape, (10, 0, 3))
        assert_array_equal(p.shape, (10, 0, 3))
        f, p = csd(np.empty((10, 0, 1)), np.zeros((10, 10, 3)), axis=1)
        assert_array_equal(f.shape, (10, 0, 3))
        assert_array_equal(p.shape, (10, 0, 3))

    def test_short_data(self):
        x = np.zeros(8)
        x[0] = 1
        with suppress_warnings() as sup:
            msg = 'nperseg = 256 is greater than input length  = 8, using nperseg = 8'
            sup.filter(UserWarning, msg)
            f, p = csd(x, x, window='hann')
            f1, p1 = csd(x, x, window='hann', nperseg=256)
        f2, p2 = csd(x, x, nperseg=8)
        assert_allclose(f, f2)
        assert_allclose(p, p2)
        assert_allclose(f1, f2)
        assert_allclose(p1, p2)

    def test_window_long_or_nd(self):
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1, np.array([1, 1, 1, 1, 1]))
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1, np.arange(6).reshape((2, 3)))

    def test_nondefault_noverlap(self):
        x = np.zeros(64)
        x[::8] = 1
        f, p = csd(x, x, nperseg=16, noverlap=4)
        q = np.array([0, 1.0 / 12.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 6.0])
        assert_allclose(p, q, atol=1e-12)

    def test_bad_noverlap(self):
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1, 'hann', 2, 7)

    def test_nfft_too_short(self):
        assert_raises(ValueError, csd, np.ones(12), np.zeros(12), nfft=3, nperseg=4)

    def test_real_onesided_even_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222, 0.11111111], 'f')
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)
        assert_(p.dtype == q.dtype)

    def test_real_onesided_odd_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=9)
        assert_allclose(f, np.arange(5.0) / 9.0)
        q = np.array([0.12477458, 0.23430935, 0.17072113, 0.17072116, 0.17072113], 'f')
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)
        assert_(p.dtype == q.dtype)

    def test_real_twosided_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.07638889], 'f')
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)
        assert_(p.dtype == q.dtype)

    def test_complex_32(self):
        x = np.zeros(16, 'F')
        x[0] = 1.0 + 2j
        x[8] = 1.0 + 2j
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.41666666, 0.38194442, 0.55555552, 0.55555552, 0.55555558, 0.55555552, 0.55555552, 0.38194442], 'f')
        assert_allclose(p, q, atol=1e-07, rtol=1e-07)
        assert_(p.dtype == q.dtype, f'dtype mismatch, {p.dtype}, {q.dtype}')

    def test_padded_freqs(self):
        x = np.zeros(12)
        y = np.ones(12)
        nfft = 24
        f = fftfreq(nfft, 1.0)[:nfft // 2 + 1]
        f[-1] *= -1
        fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
        feven, _ = csd(x, y, nperseg=6, nfft=nfft)
        assert_allclose(f, fodd)
        assert_allclose(f, feven)
        nfft = 25
        f = fftfreq(nfft, 1.0)[:(nfft + 1) // 2]
        fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
        feven, _ = csd(x, y, nperseg=6, nfft=nfft)
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

    def test_copied_data(self):
        x = np.random.randn(64)
        y = x.copy()
        _, p_same = csd(x, x, nperseg=8, average='mean', return_onesided=False)
        _, p_copied = csd(x, y, nperseg=8, average='mean', return_onesided=False)
        assert_allclose(p_same, p_copied)
        _, p_same = csd(x, x, nperseg=8, average='median', return_onesided=False)
        _, p_copied = csd(x, y, nperseg=8, average='median', return_onesided=False)
        assert_allclose(p_same, p_copied)