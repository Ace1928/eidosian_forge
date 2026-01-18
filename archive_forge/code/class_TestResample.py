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
class TestResample:

    def test_basic(self):
        sig = np.arange(128)
        num = 256
        win = signal.get_window(('kaiser', 8.0), 160)
        assert_raises(ValueError, signal.resample, sig, num, window=win)
        assert_raises(ValueError, signal.resample_poly, sig, 'yo', 1)
        assert_raises(ValueError, signal.resample_poly, sig, 1, 0)
        assert_raises(ValueError, signal.resample_poly, sig, 2, 1, padtype='')
        assert_raises(ValueError, signal.resample_poly, sig, 2, 1, padtype='mean', cval=10)
        sig2 = np.tile(np.arange(160), (2, 1))
        signal.resample(sig2, num, axis=-1, window=win)
        assert_(win.shape == (160,))

    @pytest.mark.parametrize('window', (None, 'hamming'))
    @pytest.mark.parametrize('N', (20, 19))
    @pytest.mark.parametrize('num', (100, 101, 10, 11))
    def test_rfft(self, N, num, window):
        x = np.linspace(0, 10, N, endpoint=False)
        y = np.cos(-x ** 2 / 6.0)
        assert_allclose(signal.resample(y, num, window=window), signal.resample(y + 0j, num, window=window).real)
        y = np.array([np.cos(-x ** 2 / 6.0), np.sin(-x ** 2 / 6.0)])
        y_complex = y + 0j
        assert_allclose(signal.resample(y, num, axis=1, window=window), signal.resample(y_complex, num, axis=1, window=window).real, atol=1e-09)

    def test_input_domain(self):
        tsig = np.arange(256) + 0j
        fsig = fft(tsig)
        num = 256
        assert_allclose(signal.resample(fsig, num, domain='freq'), signal.resample(tsig, num, domain='time'), atol=1e-09)

    @pytest.mark.parametrize('nx', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('ny', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('dtype', ('float', 'complex'))
    def test_dc(self, nx, ny, dtype):
        x = np.array([1] * nx, dtype)
        y = signal.resample(x, ny)
        assert_allclose(y, [1] * ny)

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_mutable_window(self, padtype):
        impulse = np.zeros(3)
        window = np.random.RandomState(0).randn(2)
        window_orig = window.copy()
        signal.resample_poly(impulse, 5, 1, window=window, padtype=padtype)
        assert_array_equal(window, window_orig)

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_output_float32(self, padtype):
        x = np.arange(10, dtype=np.float32)
        h = np.array([1, 1, 1], dtype=np.float32)
        y = signal.resample_poly(x, 1, 2, window=h, padtype=padtype)
        assert y.dtype == np.float32

    @pytest.mark.parametrize('padtype', padtype_options)
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_output_match_dtype(self, padtype, dtype):
        x = np.arange(10, dtype=dtype)
        y = signal.resample_poly(x, 1, 2, padtype=padtype)
        assert y.dtype == x.dtype

    @pytest.mark.parametrize('method, ext, padtype', [('fft', False, None)] + list(product(['polyphase'], [False, True], padtype_options)))
    def test_resample_methods(self, method, ext, padtype):
        rate = 100
        rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]
        t = np.arange(rate) / float(rate)
        freqs = np.array((1.0, 10.0, 40.0))[:, np.newaxis]
        x = np.sin(2 * np.pi * freqs * t) * hann(rate)
        for rate_to in rates_to:
            t_to = np.arange(rate_to) / float(rate_to)
            y_tos = np.sin(2 * np.pi * freqs * t_to) * hann(rate_to)
            if method == 'fft':
                y_resamps = signal.resample(x, rate_to, axis=-1)
            else:
                if ext and rate_to != rate:
                    g = gcd(rate_to, rate)
                    up = rate_to // g
                    down = rate // g
                    max_rate = max(up, down)
                    f_c = 1.0 / max_rate
                    half_len = 10 * max_rate
                    window = signal.firwin(2 * half_len + 1, f_c, window=('kaiser', 5.0))
                    polyargs = {'window': window, 'padtype': padtype}
                else:
                    polyargs = {'padtype': padtype}
                y_resamps = signal.resample_poly(x, rate_to, rate, axis=-1, **polyargs)
            for y_to, y_resamp, freq in zip(y_tos, y_resamps, freqs):
                if freq >= 0.5 * rate_to:
                    y_to.fill(0.0)
                    if padtype in ['minimum', 'maximum']:
                        assert_allclose(y_resamp, y_to, atol=0.3)
                    else:
                        assert_allclose(y_resamp, y_to, atol=0.001)
                else:
                    assert_array_equal(y_to.shape, y_resamp.shape)
                    corr = np.corrcoef(y_to, y_resamp)[0, 1]
                    assert_(corr > 0.99, msg=(corr, rate, rate_to))
        rng = np.random.RandomState(0)
        x = hann(rate) * np.cumsum(rng.randn(rate))
        for rate_to in rates_to:
            t_to = np.arange(rate_to) / float(rate_to)
            y_to = np.interp(t_to, t, x)
            if method == 'fft':
                y_resamp = signal.resample(x, rate_to)
            else:
                y_resamp = signal.resample_poly(x, rate_to, rate, padtype=padtype)
            assert_array_equal(y_to.shape, y_resamp.shape)
            corr = np.corrcoef(y_to, y_resamp)[0, 1]
            assert_(corr > 0.99, msg=corr)
        if method == 'fft':
            x1 = np.array([1.0 + 0j, 0.0 + 0j])
            y1_test = signal.resample(x1, 4)
            y1_true = np.array([1.0 + 0j, 0.5 + 0j, 0.0 + 0j, 0.5 + 0j])
            assert_allclose(y1_test, y1_true, atol=1e-12)
            x2 = np.array([1.0, 0.5, 0.0, 0.5])
            y2_test = signal.resample(x2, 2)
            y2_true = np.array([1.0, 0.0])
            assert_allclose(y2_test, y2_true, atol=1e-12)

    def test_poly_vs_filtfilt(self):
        random_state = np.random.RandomState(17)
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]
        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)
            x[0] = 0
            x[-1] = 0
            for down in down_factors:
                h = signal.firwin(31, 1.0 / down, window='hamming')
                yf = filtfilt(h, 1.0, x, padtype='constant')[::down]
                hc = convolve(h, h[::-1])
                y = signal.resample_poly(x, 1, down, window=hc)
                assert_allclose(yf, y, atol=1e-07, rtol=1e-07)

    def test_correlate1d(self):
        for down in [2, 4]:
            for nx in range(1, 40, down):
                for nweights in (32, 33):
                    x = np.random.random((nx,))
                    weights = np.random.random((nweights,))
                    y_g = correlate1d(x, weights[::-1], mode='constant')
                    y_s = signal.resample_poly(x, up=1, down=down, window=weights)
                    assert_allclose(y_g[::down], y_s)