import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
class TestFFTShift:

    @array_api_compatible
    def test_fft_n(self, xp):
        x = xp.asarray([1, 2, 3])
        if xp.__name__ == 'torch':
            assert_raises(RuntimeError, fft.fft, x, 0)
        else:
            assert_raises(ValueError, fft.fft, x, 0)