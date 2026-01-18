import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestBartHann:

    def test_basic(self):
        assert_allclose(windows.barthann(6, sym=True), [0, 0.35857354213752, 0.8794264578624801, 0.8794264578624801, 0.3585735421375199, 0], rtol=1e-15, atol=1e-15)
        assert_allclose(windows.barthann(7), [0, 0.27, 0.73, 1.0, 0.73, 0.27, 0], rtol=1e-15, atol=1e-15)
        assert_allclose(windows.barthann(6, False), [0, 0.27, 0.73, 1.0, 0.73, 0.27], rtol=1e-15, atol=1e-15)