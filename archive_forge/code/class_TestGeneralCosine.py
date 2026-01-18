import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestGeneralCosine:

    def test_basic(self):
        assert_allclose(windows.general_cosine(5, [0.5, 0.3, 0.2]), [0.4, 0.3, 1, 0.3, 0.4])
        assert_allclose(windows.general_cosine(4, [0.5, 0.3, 0.2], sym=False), [0.4, 0.3, 1, 0.3])