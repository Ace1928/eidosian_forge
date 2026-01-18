import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
class TestRemez:

    def test_bad_args(self):
        assert_raises(ValueError, remez, 11, [0.1, 0.4], [1], type='pooka')

    def test_hilbert(self):
        N = 11
        a = 0.1
        h = remez(11, [a, 0.5 - a], [1], type='hilbert')
        assert_(len(h) == N, 'Number of Taps')
        assert_array_almost_equal(h[:(N - 1) // 2], -h[:-(N - 1) // 2 - 1:-1])
        assert_((abs(h[1::2]) < 1e-15).all(), 'Even Coefficients Equal Zero')
        w, H = freqz(h, 1)
        f = w / 2 / np.pi
        Hmag = abs(H)
        assert_((Hmag[[0, -1]] < 0.02).all(), 'Zero at zero and pi')
        idx = np.logical_and(f > a, f < 0.5 - a)
        assert_((abs(Hmag[idx] - 1) < 0.015).all(), 'Pass Band Close To Unity')

    def test_compare(self):
        k = [0.02459027051844, -0.041314581814658, -0.075943803756711, -0.00353091123104, 0.193140296954975, 0.373400753484939, 0.373400753484939, 0.193140296954975, -0.00353091123104, -0.075943803756711, -0.041314581814658, 0.02459027051844]
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "'remez'")
            h = remez(12, [0, 0.3, 0.5, 1], [1, 0], Hz=2.0)
        assert_allclose(h, k)
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.0)
        assert_allclose(h, k)
        h = [-0.038976016082299, 0.018704846485491, -0.014644062687875, 0.002879152556419, 0.01684997852815, -0.043276706138248, 0.073641298245579, -0.103908158578635, 0.129770906801075, -0.147163447297124, 0.153302248456347, -0.147163447297124, 0.129770906801075, -0.103908158578635, 0.073641298245579, -0.043276706138248, 0.01684997852815, 0.002879152556419, -0.014644062687875, 0.018704846485491, -0.038976016082299]
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "'remez'")
            assert_allclose(remez(21, [0, 0.8, 0.9, 1], [0, 1], Hz=2.0), h)
        assert_allclose(remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.0), h)

    def test_remez_deprecations(self):
        with pytest.deprecated_call(match="'remez' keyword argument 'Hz'"):
            remez(12, [0, 0.3, 0.5, 1], [1, 0], Hz=2.0)
        with pytest.deprecated_call(match='use keyword arguments'):
            remez(11, [0.1, 0.4], [1], None)