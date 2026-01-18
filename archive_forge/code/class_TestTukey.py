import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestTukey:

    def test_basic(self):
        for k, v in tukey_data.items():
            if v is None:
                assert_raises(ValueError, windows.tukey, *k)
            else:
                win = windows.tukey(*k)
                assert_allclose(win, v, rtol=1e-15, atol=1e-15)

    def test_extremes(self):
        tuk0 = windows.tukey(100, 0)
        box0 = windows.boxcar(100)
        assert_array_almost_equal(tuk0, box0)
        tuk1 = windows.tukey(100, 1)
        han1 = windows.hann(100)
        assert_array_almost_equal(tuk1, han1)