import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
class TestFirls:

    def test_bad_args(self):
        assert_raises(ValueError, firls, 10, [0.1, 0.2], [0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.4], [0, 0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.4], [0, 0, 0])
        assert_raises(ValueError, firls, 11, [0.2, 0.1], [0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.3], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.3, 0.4, 0.1, 0.2], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.1, 0.3, 0.2, 0.4], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [-1, 1])
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], weight=[1, 2])
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], weight=[-1])

    def test_firls(self):
        N = 11
        a = 0.1
        h = firls(11, [0, a, 0.5 - a, 0.5], [1, 1, 0, 0], fs=1.0)
        assert_equal(len(h), N)
        midx = (N - 1) // 2
        assert_array_almost_equal(h[:midx], h[:-midx - 1:-1])
        assert_almost_equal(h[midx], 0.5)
        hodd = np.hstack((h[1:midx:2], h[-midx + 1::2]))
        assert_array_almost_equal(hodd, 0)
        w, H = freqz(h, 1)
        f = w / 2 / np.pi
        Hmag = np.abs(H)
        idx = np.logical_and(f > 0, f < a)
        assert_array_almost_equal(Hmag[idx], 1, decimal=3)
        idx = np.logical_and(f > 0.5 - a, f < 0.5)
        assert_array_almost_equal(Hmag[idx], 0, decimal=3)

    def test_compare(self):
        taps = firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], weight=[1, 2])
        known_taps = [-0.000626930101730182, -0.103354450635036, -0.00981576747564301, 0.317271686090449, 0.511409425599933, 0.317271686090449, -0.00981576747564301, -0.103354450635036, -0.000626930101730182]
        assert_allclose(taps, known_taps)
        taps = firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], weight=[1, 2])
        known_taps = [0.058545300496815, -0.014233383714318, -0.104688258464392, 0.012403323025279, 0.317930861136062, 0.4880472200297, 0.317930861136062, 0.012403323025279, -0.104688258464392, -0.014233383714318, 0.058545300496815]
        assert_allclose(taps, known_taps)
        taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], fs=20)
        known_taps = [1.156090832768218, -4.138589472739585, 7.528861916432183, -8.553057259294786, 7.528861916432183, -4.138589472739585, 1.156090832768218]
        assert_allclose(taps, known_taps)
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
            taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], nyq=10)
            assert_allclose(taps, known_taps)
            with pytest.raises(ValueError, match='between 0 and 1'):
                firls(7, [0, 1], [0, 1], nyq=0.5)

    def test_rank_deficient(self):
        x = firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.0)
        assert_allclose(np.abs(h[:2]), 1.0, atol=1e-05)
        assert_allclose(np.abs(h[-2:]), 0.0, atol=1e-06)
        x = firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.0)
        mask = w < 0.01
        assert mask.sum() > 3
        assert_allclose(np.abs(h[mask]), 1.0, atol=0.0001)
        mask = w > 0.99
        assert mask.sum() > 3
        assert_allclose(np.abs(h[mask]), 0.0, atol=0.0001)

    def test_firls_deprecations(self):
        with pytest.deprecated_call(match="argument 'nyq' is deprecated"):
            firls(1, (0, 1), (0, 0), nyq=10)
        with pytest.deprecated_call(match='use keyword arguments'):
            firls(11, [0, 0.1, 0.4, 0.5], [1, 1, 0, 0], None)