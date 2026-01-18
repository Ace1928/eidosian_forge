import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestIIRComb:

    def test_invalid_input(self):
        fs = 1000
        for args in [(-fs, 30), (0, 35), (fs / 2, 40), (fs, 35)]:
            with pytest.raises(ValueError, match='w0 must be between '):
                iircomb(*args, fs=fs)
        for args in [(120, 30), (157, 35)]:
            with pytest.raises(ValueError, match='fs must be divisible '):
                iircomb(*args, fs=fs)
        with pytest.raises(ValueError, match='fs must be divisible '):
            iircomb(w0=49.999 / int(44100 / 2), Q=30)
        with pytest.raises(ValueError, match='fs must be divisible '):
            iircomb(w0=49.999, Q=30, fs=44100)
        for args in [(0.2, 30, 'natch'), (0.5, 35, 'comb')]:
            with pytest.raises(ValueError, match='ftype must be '):
                iircomb(*args)

    @pytest.mark.parametrize('ftype', ('notch', 'peak'))
    def test_frequency_response(self, ftype):
        b, a = iircomb(1000, 30, ftype=ftype, fs=10000)
        freqs, response = freqz(b, a, 1000, fs=10000)
        comb_points = argrelextrema(abs(response), np.less)[0]
        comb1 = comb_points[0]
        assert_allclose(freqs[comb1], 1000)

    @pytest.mark.parametrize('ftype,pass_zero,peak,notch', [('peak', True, 123.45, 61.725), ('peak', False, 61.725, 123.45), ('peak', None, 61.725, 123.45), ('notch', None, 61.725, 123.45), ('notch', True, 123.45, 61.725), ('notch', False, 61.725, 123.45)])
    def test_pass_zero(self, ftype, pass_zero, peak, notch):
        b, a = iircomb(123.45, 30, ftype=ftype, fs=1234.5, pass_zero=pass_zero)
        freqs, response = freqz(b, a, [peak, notch], fs=1234.5)
        assert abs(response[0]) > 0.99
        assert abs(response[1]) < 1e-10

    def test_iir_symmetry(self):
        b, a = iircomb(400, 30, fs=24000)
        z, p, k = tf2zpk(b, a)
        assert_array_equal(sorted(z), sorted(z.conj()))
        assert_array_equal(sorted(p), sorted(p.conj()))
        assert_equal(k, np.real(k))
        assert issubclass(b.dtype.type, np.floating)
        assert issubclass(a.dtype.type, np.floating)

    def test_ba_output(self):
        b_notch, a_notch = iircomb(60, 35, ftype='notch', fs=600)
        b_notch2 = [0.957020174408697, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.957020174408697]
        a_notch2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.914040348817395]
        assert_allclose(b_notch, b_notch2)
        assert_allclose(a_notch, a_notch2)
        b_peak, a_peak = iircomb(60, 35, ftype='peak', fs=600)
        b_peak2 = [0.0429798255913026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0429798255913026]
        a_peak2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.914040348817395]
        assert_allclose(b_peak, b_peak2)
        assert_allclose(a_peak, a_peak2)

    def test_nearest_divisor(self):
        b, a = iircomb(50 / int(44100 / 2), 50.0, ftype='notch')
        freqs, response = freqz(b, a, [22000], fs=44100)
        assert abs(response[0]) < 1e-10