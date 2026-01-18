from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class MockSeries:

    def __init__(self, data):
        self.data = np.asarray(data)

    def __getattr__(self, item):
        try:
            return getattr(self.data, item)
        except AttributeError as e:
            raise AttributeError(f"'MockSeries' object has no attribute '{item}'") from e