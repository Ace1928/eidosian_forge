import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestGaussian:

    def test_basic(self):
        assert_allclose(windows.gaussian(6, 1.0), [0.04393693362340742, 0.3246524673583497, 0.8824969025845955, 0.8824969025845955, 0.3246524673583497, 0.04393693362340742])
        assert_allclose(windows.gaussian(7, 1.2), [0.04393693362340742, 0.2493522087772962, 0.7066482778577162, 1.0, 0.7066482778577162, 0.2493522087772962, 0.04393693362340742])
        assert_allclose(windows.gaussian(7, 3), [0.6065306597126334, 0.8007374029168081, 0.9459594689067654, 1.0, 0.9459594689067654, 0.8007374029168081, 0.6065306597126334])
        assert_allclose(windows.gaussian(6, 3, False), [0.6065306597126334, 0.8007374029168081, 0.9459594689067654, 1.0, 0.9459594689067654, 0.8007374029168081])