import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
class TestKaiserBesselDerived:

    def test_basic(self):
        M = 100
        w = windows.kaiser_bessel_derived(M, beta=4.0)
        w2 = windows.get_window(('kaiser bessel derived', 4.0), M, fftbins=False)
        assert_allclose(w, w2)
        assert_allclose(w[:M // 2] ** 2 + w[-M // 2:] ** 2, 1.0)
        assert_allclose(windows.kaiser_bessel_derived(2, beta=np.pi / 2)[:1], np.sqrt(2) / 2)
        assert_allclose(windows.kaiser_bessel_derived(4, beta=np.pi / 2)[:2], [0.518562710536, 0.85503959864])
        assert_allclose(windows.kaiser_bessel_derived(6, beta=np.pi / 2)[:3], [0.436168993154, 0.707106781187, 0.899864772847])

    def test_exceptions(self):
        M = 100
        msg = 'Kaiser-Bessel Derived windows are only defined for even number of points'
        with assert_raises(ValueError, match=msg):
            windows.kaiser_bessel_derived(M + 1, beta=4.0)
        msg = 'Kaiser-Bessel Derived windows are only defined for symmetric shapes'
        with assert_raises(ValueError, match=msg):
            windows.kaiser_bessel_derived(M + 1, beta=4.0, sym=False)