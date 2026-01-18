import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestNuttall:

    def test_basic(self):
        assert_allclose(windows.nuttall(6, sym=False), [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298, 0.0613345])
        assert_allclose(windows.nuttall(7, sym=False), [0.0003628, 0.03777576895352025, 0.3427276199688195, 0.8918518610776603, 0.8918518610776603, 0.3427276199688196, 0.0377757689535203])
        assert_allclose(windows.nuttall(6), [0.0003628, 0.1105152530498718, 0.7982580969501282, 0.7982580969501283, 0.1105152530498719, 0.0003628])
        assert_allclose(windows.nuttall(7, True), [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298, 0.0613345, 0.0003628])