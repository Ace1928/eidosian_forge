import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestCheb2ord:

    def test_lowpass(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'lp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)
        assert_equal(N, 8)
        assert_allclose(Wn, 0.28647639976553163, rtol=1e-15)

    def test_highpass(self):
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'hp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)
        assert_equal(N, 9)
        assert_allclose(Wn, 0.20697492182903282, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'bp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]), -rs + 0.1)
        assert_equal(N, 9)
        assert_allclose(Wn, [0.1487693756592348, 0.5974844784235148], rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'bs', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]), -rs + 0.1)
        assert_equal(N, 10)
        assert_allclose(Wn, [0.19926249974781743, 0.5012524658556736], rtol=1e-06)

    def test_analog(self):
        wp = [20, 50]
        ws = [10, 60]
        rp = 3
        rs = 80
        N, Wn = cheb2ord(wp, ws, rp, rs, True)
        b, a = cheby2(N, rs, Wn, 'bp', True)
        w, h = freqs(b, a)
        assert_array_less(-rp - 0.1, dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]), -rs + 0.1)
        assert_equal(N, 11)
        assert_allclose(Wn, [16.73740595370124, 59.74641487254268], rtol=1e-15)

    def test_fs_param(self):
        wp = 150
        ws = 100
        rp = 3
        rs = 70
        fs = 1000
        N, Wn = cheb2ord(wp, ws, rp, rs, False, fs=fs)
        b, a = cheby2(N, rs, Wn, 'hp', False, fs=fs)
        w, h = freqz(b, a, fs=fs)
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)
        assert_equal(N, 9)
        assert_allclose(Wn, 103.4874609145164, rtol=1e-15)

    def test_invalid_input(self):
        with pytest.raises(ValueError) as exc_info:
            cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 2)
        assert 'gpass should be smaller than gstop' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            cheb2ord([0.1, 0.6], [0.2, 0.5], -1, 2)
        assert 'gpass should be larger than 0.0' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            cheb2ord([0.1, 0.6], [0.2, 0.5], 1, -2)
        assert 'gstop should be larger than 0.0' in str(exc_info.value)

    def test_ellip_cheb2(self):
        n, wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        assert n == 7
        n1, w1 = cheb1ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        assert not (wn == w1).all()