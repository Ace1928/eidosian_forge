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
class TestDecimate:

    def test_bad_args(self):
        x = np.arange(12)
        assert_raises(TypeError, signal.decimate, x, q=0.5, n=1)
        assert_raises(TypeError, signal.decimate, x, q=2, n=0.5)

    def test_basic_IIR(self):
        x = np.arange(12)
        y = signal.decimate(x, 2, n=1, ftype='iir', zero_phase=False).round()
        assert_array_equal(y, x[::2])

    def test_basic_FIR(self):
        x = np.arange(12)
        y = signal.decimate(x, 2, n=1, ftype='fir', zero_phase=False).round()
        assert_array_equal(y, x[::2])

    def test_shape(self):
        z = np.zeros((30, 30))
        d0 = signal.decimate(z, 2, axis=0, zero_phase=False)
        assert_equal(d0.shape, (15, 30))
        d1 = signal.decimate(z, 2, axis=1, zero_phase=False)
        assert_equal(d1.shape, (30, 15))

    def test_phaseshift_FIR(self):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients, 'Badly conditioned filter')
            self._test_phaseshift(method='fir', zero_phase=False)

    def test_zero_phase_FIR(self):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients, 'Badly conditioned filter')
            self._test_phaseshift(method='fir', zero_phase=True)

    def test_phaseshift_IIR(self):
        self._test_phaseshift(method='iir', zero_phase=False)

    def test_zero_phase_IIR(self):
        self._test_phaseshift(method='iir', zero_phase=True)

    def _test_phaseshift(self, method, zero_phase):
        rate = 120
        rates_to = [15, 20, 30, 40]
        t_tot = 100
        t = np.arange(rate * t_tot + 1) / float(rate)
        freqs = np.array(rates_to) * 0.8 / 2
        d = np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t) * signal.windows.tukey(t.size, 0.1)
        for rate_to in rates_to:
            q = rate // rate_to
            t_to = np.arange(rate_to * t_tot + 1) / float(rate_to)
            d_tos = np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t_to) * signal.windows.tukey(t_to.size, 0.1)
            if method == 'fir':
                n = 30
                system = signal.dlti(signal.firwin(n + 1, 1.0 / q, window='hamming'), 1.0)
            elif method == 'iir':
                n = 8
                wc = 0.8 * np.pi / q
                system = signal.dlti(*signal.cheby1(n, 0.05, wc / np.pi))
            if zero_phase is False:
                _, h_resps = signal.freqz(system.num, system.den, freqs / rate * 2 * np.pi)
                h_resps /= np.abs(h_resps)
            else:
                h_resps = np.ones_like(freqs)
            y_resamps = signal.decimate(d.real, q, n, ftype=system, zero_phase=zero_phase)
            h_resamps = np.sum(d_tos.conj() * y_resamps, axis=-1)
            h_resamps /= np.abs(h_resamps)
            subnyq = freqs < 0.5 * rate_to
            assert_allclose(np.angle(h_resps.conj() * h_resamps)[subnyq], 0, atol=0.001, rtol=0.001)

    def test_auto_n(self):
        sfreq = 100.0
        n = 1000
        t = np.arange(n) / sfreq
        x = np.sqrt(2.0 / n) * np.sin(2 * np.pi * (sfreq / 30.0) * t)
        assert_allclose(np.linalg.norm(x), 1.0, rtol=0.001)
        x_out = signal.decimate(x, 30, ftype='fir')
        assert_array_less(np.linalg.norm(x_out), 0.01)

    def test_long_float32(self):
        x = signal.decimate(np.ones(10000, dtype=np.float32), 10)
        assert not any(np.isnan(x))

    def test_float16_upcast(self):
        x = signal.decimate(np.ones(100, dtype=np.float16), 10)
        assert x.dtype.type == np.float64

    def test_complex_iir_dlti(self):
        fcentre = 50
        fwidth = 5
        fs = 1000.0
        z, p, k = signal.butter(2, 2 * np.pi * fwidth / 2, output='zpk', fs=fs)
        z = z.astype(complex) * np.exp(2j * np.pi * fcentre / fs)
        p = p.astype(complex) * np.exp(2j * np.pi * fcentre / fs)
        system = signal.dlti(z, p, k)
        t = np.arange(200) / fs
        u = np.exp(2j * np.pi * fcentre * t) + 0.5 * np.exp(-2j * np.pi * fcentre * t)
        ynzp = signal.decimate(u, 2, ftype=system, zero_phase=False)
        ynzpref = signal.lfilter(*signal.zpk2tf(z, p, k), u)[::2]
        assert_equal(ynzp, ynzpref)
        yzp = signal.decimate(u, 2, ftype=system, zero_phase=True)
        yzpref = signal.filtfilt(*signal.zpk2tf(z, p, k), u)[::2]
        assert_allclose(yzp, yzpref, rtol=1e-10, atol=1e-13)

    def test_complex_fir_dlti(self):
        fcentre = 50
        fwidth = 5
        fs = 1000.0
        numtaps = 20
        bbase = signal.firwin(numtaps, fwidth / 2, fs=fs)
        zbase = np.roots(bbase)
        zrot = zbase * np.exp(2j * np.pi * fcentre / fs)
        bz = bbase[0] * np.poly(zrot)
        system = signal.dlti(bz, 1)
        t = np.arange(200) / fs
        u = np.exp(2j * np.pi * fcentre * t) + 0.5 * np.exp(-2j * np.pi * fcentre * t)
        ynzp = signal.decimate(u, 2, ftype=system, zero_phase=False)
        ynzpref = signal.upfirdn(bz, u, up=1, down=2)[:100]
        assert_equal(ynzp, ynzpref)
        yzp = signal.decimate(u, 2, ftype=system, zero_phase=True)
        yzpref = signal.resample_poly(u, 1, 2, window=bz)
        assert_equal(yzp, yzpref)