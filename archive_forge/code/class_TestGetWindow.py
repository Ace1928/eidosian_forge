import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestGetWindow:

    def test_boxcar(self):
        w = windows.get_window('boxcar', 12)
        assert_array_equal(w, np.ones_like(w))
        w = windows.get_window(('boxcar',), 16)
        assert_array_equal(w, np.ones_like(w))

    def test_cheb_odd(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            w = windows.get_window(('chebwin', -40), 53, fftbins=False)
        assert_array_almost_equal(w, cheb_odd_true, decimal=4)

    def test_cheb_even(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'This window is not suitable')
            w = windows.get_window(('chebwin', 40), 54, fftbins=False)
        assert_array_almost_equal(w, cheb_even_true, decimal=4)

    def test_dpss(self):
        win1 = windows.get_window(('dpss', 3), 64, fftbins=False)
        win2 = windows.dpss(64, 3)
        assert_array_almost_equal(win1, win2, decimal=4)

    def test_kaiser_float(self):
        win1 = windows.get_window(7.2, 64)
        win2 = windows.kaiser(64, 7.2, False)
        assert_allclose(win1, win2)

    def test_invalid_inputs(self):
        assert_raises(ValueError, windows.get_window, set('hann'), 8)
        assert_raises(ValueError, windows.get_window, 'broken', 4)

    def test_array_as_window(self):
        osfactor = 128
        sig = np.arange(128)
        win = windows.get_window(('kaiser', 8.0), osfactor // 2)
        with assert_raises(ValueError, match='must have the same length'):
            resample(sig, len(sig) * osfactor, window=win)

    def test_general_cosine(self):
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4), [0.4, 0.3, 1, 0.3])
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4, fftbins=False), [0.4, 0.55, 0.55, 0.4])

    def test_general_hamming(self):
        assert_allclose(get_window(('general_hamming', 0.7), 5), [0.4, 0.6072949, 0.9427051, 0.9427051, 0.6072949])
        assert_allclose(get_window(('general_hamming', 0.7), 5, fftbins=False), [0.4, 0.7, 1.0, 0.7, 0.4])

    def test_lanczos(self):
        assert_allclose(get_window('lanczos', 6), [0.0, 0.413496672, 0.826993343, 1.0, 0.826993343, 0.413496672], atol=1e-09)
        assert_allclose(get_window('lanczos', 6, fftbins=False), [0.0, 0.504551152, 0.935489284, 0.935489284, 0.504551152, 0.0], atol=1e-09)
        assert_allclose(get_window('lanczos', 6), get_window('sinc', 6))