import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestIIRPeak:

    def test_ba_output(self):
        b, a = iirpeak(0.06, 30)
        b2 = [0.003131764229, 0, -0.003131764229]
        a2 = [1.0, -1.958421917, 0.99373647]
        assert_allclose(b, b2, rtol=1e-08)
        assert_allclose(a, a2, rtol=1e-08)

    def test_frequency_response(self):
        b, a = iirpeak(0.3, 30)
        w, h = freqz(b, a, 1000)
        p = [30, 295, 300, 305, 800]
        hp = h[p]
        assert_allclose(abs(hp[0]), 0, atol=0.01)
        assert_allclose(abs(hp[4]), 0, atol=0.01)
        assert_allclose(abs(hp[1]), 1 / np.sqrt(2), rtol=0.01)
        assert_allclose(abs(hp[3]), 1 / np.sqrt(2), rtol=0.01)
        assert_allclose(abs(hp[2]), 1, rtol=1e-10)

    def test_errors(self):
        assert_raises(ValueError, iirpeak, w0=2, Q=30)
        assert_raises(ValueError, iirpeak, w0=-1, Q=30)
        assert_raises(ValueError, iirpeak, w0='blabla', Q=30)
        assert_raises(TypeError, iirpeak, w0=-1, Q=[1, 2, 3])

    def test_fs_param(self):
        b, a = iirpeak(1200, 30, fs=8000)
        w, h = freqz(b, a, 1000, fs=8000)
        p = [30, 295, 300, 305, 800]
        hp = h[p]
        assert_allclose(abs(hp[0]), 0, atol=0.01)
        assert_allclose(abs(hp[4]), 0, atol=0.01)
        assert_allclose(abs(hp[1]), 1 / np.sqrt(2), rtol=0.01)
        assert_allclose(abs(hp[3]), 1 / np.sqrt(2), rtol=0.01)
        assert_allclose(abs(hp[2]), 1, rtol=1e-10)