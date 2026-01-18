import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestGeneralHamming:

    def test_basic(self):
        assert_allclose(windows.general_hamming(5, 0.7), [0.4, 0.7, 1.0, 0.7, 0.4])
        assert_allclose(windows.general_hamming(5, 0.75, sym=False), [0.5, 0.6727457514, 0.9522542486, 0.9522542486, 0.6727457514])
        assert_allclose(windows.general_hamming(6, 0.75, sym=True), [0.5, 0.6727457514, 0.9522542486, 0.9522542486, 0.6727457514, 0.5])