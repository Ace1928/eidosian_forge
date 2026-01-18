import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestHann:

    def test_basic(self):
        assert_allclose(windows.hann(6, sym=False), [0, 0.25, 0.75, 1.0, 0.75, 0.25], rtol=1e-15, atol=1e-15)
        assert_allclose(windows.hann(7, sym=False), [0, 0.1882550990706332, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781572, 0.1882550990706332], rtol=1e-15, atol=1e-15)
        assert_allclose(windows.hann(6, True), [0, 0.3454915028125263, 0.9045084971874737, 0.9045084971874737, 0.3454915028125263, 0], rtol=1e-15, atol=1e-15)
        assert_allclose(windows.hann(7), [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0], rtol=1e-15, atol=1e-15)