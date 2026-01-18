import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestDPSS:

    def test_basic(self):
        for k, v in dpss_data.items():
            win, ratios = windows.dpss(*k, return_ratios=True)
            assert_allclose(win, v[0], atol=1e-07, err_msg=k)
            assert_allclose(ratios, v[1], rtol=1e-05, atol=1e-07, err_msg=k)

    def test_unity(self):
        for M in range(1, 21):
            win = windows.dpss(M, M / 2.1)
            expected = M % 2
            assert_equal(np.isclose(win, 1.0).sum(), expected, err_msg=f'{win}')
            win_sub = windows.dpss(M, M / 2.1, norm='subsample')
            if M > 2:
                assert_equal(np.isclose(win_sub, 1.0).sum(), expected, err_msg=f'{win_sub}')
                assert_allclose(win, win_sub, rtol=0.03)
            win_2 = windows.dpss(M, M / 2.1, norm=2)
            expected = 1 if M == 1 else 0
            assert_equal(np.isclose(win_2, 1.0).sum(), expected, err_msg=f'{win_2}')

    def test_extremes(self):
        lam = windows.dpss(31, 6, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.0)
        lam = windows.dpss(31, 7, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.0)
        lam = windows.dpss(31, 8, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.0)

    def test_degenerate(self):
        assert_raises(ValueError, windows.dpss, 4, 1.5, -1)
        assert_raises(ValueError, windows.dpss, 4, 1.5, -5)
        assert_raises(TypeError, windows.dpss, 4, 1.5, 1.1)
        assert_raises(ValueError, windows.dpss, 3, 1.5, 3)
        assert_raises(ValueError, windows.dpss, 3, -1, 3)
        assert_raises(ValueError, windows.dpss, 3, 0, 3)
        assert_raises(ValueError, windows.dpss, -1, 1, 3)