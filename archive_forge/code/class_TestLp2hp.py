import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2hp:

    def test_basic(self):
        b = [0.2505943232519002]
        a = [1, 0.5972404165413486, 0.9283480575752417, 0.2505943232519002]
        b_hp, a_hp = lp2hp(b, a, 2 * np.pi * 5000)
        assert_allclose(b_hp, [1, 0, 0, 0])
        assert_allclose(a_hp, [1, 116380.0, 2352200000.0, 123730000000000.0], rtol=0.0001)