import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestHamming:

    def test_basic(self):
        assert_allclose(windows.hamming(6, False), [0.08, 0.31, 0.77, 1.0, 0.77, 0.31])
        assert_allclose(windows.hamming(7, sym=False), [0.08, 0.2531946911449826, 0.6423596296199047, 0.9544456792351128, 0.9544456792351128, 0.6423596296199047, 0.2531946911449826])
        assert_allclose(windows.hamming(6), [0.08, 0.3978521825875242, 0.9121478174124757, 0.9121478174124757, 0.3978521825875242, 0.08])
        assert_allclose(windows.hamming(7, sym=True), [0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08])