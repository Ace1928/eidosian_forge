import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestFreqs:

    def test_basic(self):
        _, h = freqs([1.0], [1.0], worN=8)
        assert_array_almost_equal(h, np.ones(8))

    def test_output(self):
        w = [0.1, 1, 10, 100]
        num = [1]
        den = [1, 1]
        w, H = freqs(num, den, worN=w)
        s = w * 1j
        expected = 1 / (s + 1)
        assert_array_almost_equal(H.real, expected.real)
        assert_array_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        num = [1]
        den = [1, 1]
        n = 10
        expected_w = np.logspace(-2, 1, n)
        w, H = freqs(num, den, worN=n)
        assert_array_almost_equal(w, expected_w)

    def test_plot(self):

        def plot(w, h):
            assert_array_almost_equal(h, np.ones(8))
        assert_raises(ZeroDivisionError, freqs, [1.0], [1.0], worN=8, plot=lambda w, h: 1 / 0)
        freqs([1.0], [1.0], worN=8, plot=plot)

    def test_backward_compat(self):
        w1, h1 = freqs([1.0], [1.0])
        w2, h2 = freqs([1.0], [1.0], None)
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(h1, h2)

    def test_w_or_N_types(self):
        for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8), np.array(8)):
            w, h = freqs([1.0], [1.0], worN=N)
            assert_equal(len(w), 8)
            assert_array_almost_equal(h, np.ones(8))
        for w in (8.0, 8.0 + 0j):
            w_out, h = freqs([1.0], [1.0], worN=w)
            assert_array_almost_equal(w_out, [8])
            assert_array_almost_equal(h, [1])