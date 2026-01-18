from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
class TestFFTFreq:

    @skip_if_array_api_backend('numpy.array_api')
    @skip_if_array_api_backend('cupy')
    @array_api_compatible
    def test_definition(self, xp):
        device = SCIPY_DEVICE
        try:
            x = xp.asarray([0, 1, 2, 3, 4, -4, -3, -2, -1], dtype=xp.float64, device=device)
            x2 = xp.asarray([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=xp.float64, device=device)
        except TypeError:
            x = xp.asarray([0, 1, 2, 3, 4, -4, -3, -2, -1], dtype=xp.float64)
            x2 = xp.asarray([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=xp.float64)
        y = xp.asarray(9 * fft.fftfreq(9, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x)
        y = xp.asarray(9 * xp.pi * fft.fftfreq(9, xp.pi, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x)
        y = xp.asarray(10 * fft.fftfreq(10, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x2)
        y = xp.asarray(10 * xp.pi * fft.fftfreq(10, xp.pi, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x2)