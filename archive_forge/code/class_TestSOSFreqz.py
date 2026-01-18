import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestSOSFreqz:

    def test_sosfreqz_basic(self):
        N = 500
        b, a = butter(4, 0.2)
        sos = butter(4, 0.2, output='sos')
        w, h = freqz(b, a, worN=N)
        w2, h2 = sosfreqz(sos, worN=N)
        assert_equal(w2, w)
        assert_allclose(h2, h, rtol=1e-10, atol=1e-14)
        b, a = ellip(3, 1, 30, (0.2, 0.3), btype='bandpass')
        sos = ellip(3, 1, 30, (0.2, 0.3), btype='bandpass', output='sos')
        w, h = freqz(b, a, worN=N)
        w2, h2 = sosfreqz(sos, worN=N)
        assert_equal(w2, w)
        assert_allclose(h2, h, rtol=1e-10, atol=1e-14)
        assert_raises(ValueError, sosfreqz, sos[:0])

    def test_sosfrez_design(self):
        N, Wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        sos = cheby2(N, 60, Wn, 'stop', output='sos')
        w, h = sosfreqz(sos)
        h = np.abs(h)
        w /= np.pi
        assert_allclose(20 * np.log10(h[w <= 0.1]), 0, atol=3.01)
        assert_allclose(20 * np.log10(h[w >= 0.6]), 0.0, atol=3.01)
        assert_allclose(h[(w >= 0.2) & (w <= 0.5)], 0.0, atol=0.001)
        N, Wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 150)
        sos = cheby2(N, 150, Wn, 'stop', output='sos')
        w, h = sosfreqz(sos)
        dB = 20 * np.log10(np.abs(h))
        w /= np.pi
        assert_allclose(dB[w <= 0.1], 0, atol=3.01)
        assert_allclose(dB[w >= 0.6], 0.0, atol=3.01)
        assert_array_less(dB[(w >= 0.2) & (w <= 0.5)], -149.9)
        N, Wn = cheb1ord(0.2, 0.3, 3, 40)
        sos = cheby1(N, 3, Wn, 'low', output='sos')
        w, h = sosfreqz(sos)
        h = np.abs(h)
        w /= np.pi
        assert_allclose(20 * np.log10(h[w <= 0.2]), 0, atol=3.01)
        assert_allclose(h[w >= 0.3], 0.0, atol=0.01)
        N, Wn = cheb1ord(0.2, 0.3, 1, 150)
        sos = cheby1(N, 1, Wn, 'low', output='sos')
        w, h = sosfreqz(sos)
        dB = 20 * np.log10(np.abs(h))
        w /= np.pi
        assert_allclose(dB[w <= 0.2], 0, atol=1.01)
        assert_array_less(dB[w >= 0.3], -149.9)
        N, Wn = ellipord(0.3, 0.2, 3, 60)
        sos = ellip(N, 0.3, 60, Wn, 'high', output='sos')
        w, h = sosfreqz(sos)
        h = np.abs(h)
        w /= np.pi
        assert_allclose(20 * np.log10(h[w >= 0.3]), 0, atol=3.01)
        assert_allclose(h[w <= 0.1], 0.0, atol=0.0015)
        N, Wn = buttord([0.2, 0.5], [0.14, 0.6], 3, 40)
        sos = butter(N, Wn, 'band', output='sos')
        w, h = sosfreqz(sos)
        h = np.abs(h)
        w /= np.pi
        assert_allclose(h[w <= 0.14], 0.0, atol=0.01)
        assert_allclose(h[w >= 0.6], 0.0, atol=0.01)
        assert_allclose(20 * np.log10(h[(w >= 0.2) & (w <= 0.5)]), 0, atol=3.01)
        N, Wn = buttord([0.2, 0.5], [0.14, 0.6], 3, 100)
        sos = butter(N, Wn, 'band', output='sos')
        w, h = sosfreqz(sos)
        dB = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
        w /= np.pi
        assert_array_less(dB[(w > 0) & (w <= 0.14)], -99.9)
        assert_array_less(dB[w >= 0.6], -99.9)
        assert_allclose(dB[(w >= 0.2) & (w <= 0.5)], 0, atol=3.01)

    def test_sosfreqz_design_ellip(self):
        N, Wn = ellipord(0.3, 0.1, 3, 60)
        sos = ellip(N, 0.3, 60, Wn, 'high', output='sos')
        w, h = sosfreqz(sos)
        h = np.abs(h)
        w /= np.pi
        assert_allclose(20 * np.log10(h[w >= 0.3]), 0, atol=3.01)
        assert_allclose(h[w <= 0.1], 0.0, atol=0.0015)
        N, Wn = ellipord(0.3, 0.2, 0.5, 150)
        sos = ellip(N, 0.5, 150, Wn, 'high', output='sos')
        w, h = sosfreqz(sos)
        dB = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
        w /= np.pi
        assert_allclose(dB[w >= 0.3], 0, atol=0.55)
        assert dB[w <= 0.2].max() < -150 * (1 - 1e-12)

    @mpmath_check('0.10')
    def test_sos_freqz_against_mp(self):
        from . import mpsig
        N = 500
        order = 25
        Wn = 0.15
        with mpmath.workdps(80):
            z_mp, p_mp, k_mp = mpsig.butter_lp(order, Wn)
            w_mp, h_mp = mpsig.zpkfreqz(z_mp, p_mp, k_mp, N)
        w_mp = np.array([float(x) for x in w_mp])
        h_mp = np.array([complex(x) for x in h_mp])
        sos = butter(order, Wn, output='sos')
        w, h = sosfreqz(sos, worN=N)
        assert_allclose(w, w_mp, rtol=1e-12, atol=1e-14)
        assert_allclose(h, h_mp, rtol=1e-12, atol=1e-14)

    def test_fs_param(self):
        fs = 900
        sos = [[0.03934683014103762, 0.07869366028207524, 0.03934683014103762, 1.0, -0.37256600288916636, 0.0], [1.0, 1.0, 0.0, 1.0, -0.9495739996946778, 0.45125966317124144]]
        w1, h1 = sosfreqz(sos, fs=fs)
        w2, h2 = sosfreqz(sos)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs / 2, 512, endpoint=False))
        w1, h1 = sosfreqz(sos, whole=True, fs=fs)
        w2, h2 = sosfreqz(sos, whole=True)
        assert_allclose(h1, h2, atol=1e-27)
        assert_allclose(w1, np.linspace(0, fs, 512, endpoint=False))
        w1, h1 = sosfreqz(sos, 5, fs=fs)
        w2, h2 = sosfreqz(sos, 5)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs / 2, 5, endpoint=False))
        w1, h1 = sosfreqz(sos, 5, whole=True, fs=fs)
        w2, h2 = sosfreqz(sos, 5, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 5, endpoint=False))
        for w in ([123], (123,), np.array([123]), (50, 123, 230), np.array([50, 123, 230])):
            w1, h1 = sosfreqz(sos, w, fs=fs)
            w2, h2 = sosfreqz(sos, 2 * pi * np.array(w) / fs)
            assert_allclose(h1, h2)
            assert_allclose(w, w1)

    def test_w_or_N_types(self):
        for N in (7, np.int8(7), np.int16(7), np.int32(7), np.int64(7), np.array(7), 8, np.int8(8), np.int16(8), np.int32(8), np.int64(8), np.array(8)):
            w, h = sosfreqz([1, 0, 0, 1, 0, 0], worN=N)
            assert_array_almost_equal(w, np.pi * np.arange(N) / N)
            assert_array_almost_equal(h, np.ones(N))
            w, h = sosfreqz([1, 0, 0, 1, 0, 0], worN=N, fs=100)
            assert_array_almost_equal(w, np.linspace(0, 50, N, endpoint=False))
            assert_array_almost_equal(h, np.ones(N))
        for w in (8.0, 8.0 + 0j):
            w_out, h = sosfreqz([1, 0, 0, 1, 0, 0], worN=w, fs=100)
            assert_array_almost_equal(w_out, [8])
            assert_array_almost_equal(h, [1])