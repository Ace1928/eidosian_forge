import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2bs:

    def test_basic(self):
        b = [1]
        a = [1, 1]
        b_bs, a_bs = lp2bs(b, a, 0.41722257286366754, 0.1846057532615225)
        assert_array_almost_equal(b_bs, [1, 0, 0.17407], decimal=5)
        assert_array_almost_equal(a_bs, [1, 0.18461, 0.17407], decimal=5)