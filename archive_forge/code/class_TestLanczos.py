import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestLanczos:

    def test_basic(self):
        assert_allclose(windows.lanczos(6, sym=False), [0.0, 0.413496672, 0.826993343, 1.0, 0.826993343, 0.413496672], atol=1e-09)
        assert_allclose(windows.lanczos(6), [0.0, 0.504551152, 0.935489284, 0.935489284, 0.504551152, 0.0], atol=1e-09)
        assert_allclose(windows.lanczos(7, sym=True), [0.0, 0.413496672, 0.826993343, 1.0, 0.826993343, 0.413496672, 0.0], atol=1e-09)

    def test_array_size(self):
        for n in [0, 10, 11]:
            assert_equal(len(windows.lanczos(n, sym=False)), n)
            assert_equal(len(windows.lanczos(n, sym=True)), n)