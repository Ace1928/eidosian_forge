import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestSos2Zpk:

    def test_basic(self):
        sos = [[1, 0, 1, 1, 0, -0.81], [1, 0, 0, 1, 0, +0.49]]
        z, p, k = sos2zpk(sos)
        z2 = [1j, -1j, 0, 0]
        p2 = [0.9, -0.9, 0.7j, -0.7j]
        k2 = 1
        assert_array_almost_equal(sort(z), sort(z2), decimal=4)
        assert_array_almost_equal(sort(p), sort(p2), decimal=4)
        assert_array_almost_equal(k, k2)
        sos = [[1.0, +0.61803, 1.0, 1.0, +0.60515, 0.95873], [1.0, -1.61803, 1.0, 1.0, -1.5843, 0.95873], [1.0, +1.0, 0.0, 1.0, +0.97915, 0.0]]
        z, p, k = sos2zpk(sos)
        z2 = [-0.309 + 0.9511j, -0.309 - 0.9511j, 0.809 + 0.5878j, 0.809 - 0.5878j, -1.0 + 0j, 0]
        p2 = [-0.3026 + 0.9312j, -0.3026 - 0.9312j, 0.7922 + 0.5755j, 0.7922 - 0.5755j, -0.9791 + 0j, 0]
        k2 = 1
        assert_array_almost_equal(sort(z), sort(z2), decimal=4)
        assert_array_almost_equal(sort(p), sort(p2), decimal=4)
        sos = array([[1, 2, 3, 1, 0.2, 0.3], [4, 5, 6, 1, 0.4, 0.5]])
        z = array([-1 - 1.4142135623731j, -1 + 1.4142135623731j, -0.625 - 1.05326872164704j, -0.625 + 1.05326872164704j])
        p = array([-0.2 - 0.678232998312527j, -0.2 + 0.678232998312527j, -0.1 - 0.53851648071345j, -0.1 + 0.53851648071345j])
        k = 4
        z2, p2, k2 = sos2zpk(sos)
        assert_allclose(_cplxpair(z2), z)
        assert_allclose(_cplxpair(p2), p)
        assert_allclose(k2, k)

    def test_fewer_zeros(self):
        """Test not the expected number of p/z (effectively at origin)."""
        sos = butter(3, 0.1, output='sos')
        z, p, k = sos2zpk(sos)
        assert len(z) == 4
        assert len(p) == 4
        sos = butter(12, [5.0, 30.0], 'bandpass', fs=1200.0, analog=False, output='sos')
        with pytest.warns(BadCoefficients, match='Badly conditioned'):
            z, p, k = sos2zpk(sos)
        assert len(z) == 24
        assert len(p) == 24