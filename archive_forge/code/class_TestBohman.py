import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestBohman:

    def test_basic(self):
        assert_allclose(windows.bohman(6), [0, 0.1791238937062839, 0.8343114522576858, 0.8343114522576858, 0.1791238937062838, 0])
        assert_allclose(windows.bohman(7, sym=True), [0, 0.1089977810442293, 0.6089977810442293, 1.0, 0.6089977810442295, 0.1089977810442293, 0])
        assert_allclose(windows.bohman(6, False), [0, 0.1089977810442293, 0.6089977810442293, 1.0, 0.6089977810442295, 0.1089977810442293])