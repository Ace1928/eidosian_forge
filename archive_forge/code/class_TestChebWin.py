import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestChebWin:

    def test_basic(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            assert_allclose(windows.chebwin(6, 100), [0.1046401879356917, 0.5075781475823447, 1.0, 1.0, 0.5075781475823447, 0.1046401879356917])
            assert_allclose(windows.chebwin(7, 100), [0.05650405062850233, 0.316608530648474, 0.7601208123539079, 1.0, 0.7601208123539079, 0.316608530648474, 0.05650405062850233])
            assert_allclose(windows.chebwin(6, 10), [1.0, 0.6071201674458373, 0.6808391469897297, 0.6808391469897297, 0.6071201674458373, 1.0])
            assert_allclose(windows.chebwin(7, 10), [1.0, 0.5190521247588651, 0.5864059018130382, 0.6101519801307441, 0.5864059018130382, 0.5190521247588651, 1.0])
            assert_allclose(windows.chebwin(6, 10, False), [1.0, 0.5190521247588651, 0.5864059018130382, 0.6101519801307441, 0.5864059018130382, 0.5190521247588651])

    def test_cheb_odd_high_attenuation(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            cheb_odd = windows.chebwin(53, at=-40)
        assert_array_almost_equal(cheb_odd, cheb_odd_true, decimal=4)

    def test_cheb_even_high_attenuation(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            cheb_even = windows.chebwin(54, at=40)
        assert_array_almost_equal(cheb_even, cheb_even_true, decimal=4)

    def test_cheb_odd_low_attenuation(self):
        cheb_odd_low_at_true = array([1.0, 0.519052, 0.586405, 0.610151, 0.586405, 0.519052, 1.0])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            cheb_odd = windows.chebwin(7, at=10)
        assert_array_almost_equal(cheb_odd, cheb_odd_low_at_true, decimal=4)

    def test_cheb_even_low_attenuation(self):
        cheb_even_low_at_true = array([1.0, 0.451924, 0.51027, 0.541338, 0.541338, 0.51027, 0.451924, 1.0])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            cheb_even = windows.chebwin(8, at=-10)
        assert_array_almost_equal(cheb_even, cheb_even_low_at_true, decimal=4)