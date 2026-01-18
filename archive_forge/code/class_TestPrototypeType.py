import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestPrototypeType:

    def test_output_type(self):
        for func in (buttap, besselap, lambda N: cheb1ap(N, 1), lambda N: cheb2ap(N, 20), lambda N: ellipap(N, 1, 20)):
            for N in range(7):
                z, p, k = func(N)
                assert_(isinstance(z, np.ndarray))
                assert_(isinstance(p, np.ndarray))