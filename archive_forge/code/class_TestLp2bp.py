import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2bp:

    def test_basic(self):
        b = [1]
        a = [1, 2, 2, 1]
        b_bp, a_bp = lp2bp(b, a, 2 * np.pi * 4000, 2 * np.pi * 2000)
        assert_allclose(b_bp, [1984400000000.0, 0, 0, 0], rtol=1e-06)
        assert_allclose(a_bp, [1, 25133.0, 2210800000.0, 33735000000000.0, 1.3965e+18, 1.0028e+22, 2.5202e+26], rtol=0.0001)