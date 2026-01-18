import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
class TestEllipLegendreCarlsonIdentities:
    """Test identities expressing the Legendre elliptic integrals in terms
    of Carlson's symmetric integrals.  These identities can be found
    in the DLMF https://dlmf.nist.gov/19.25#i .
    """

    def setup_class(self):
        self.m_n1_1 = np.arange(-1.0, 1.0, 0.01)
        self.max_neg = finfo(double).min
        self.very_neg_m = -1.0 * 2.0 ** arange(-1 + np.log2(-self.max_neg), 0.0, -1.0)
        self.ms_up_to_1 = np.concatenate(([self.max_neg], self.very_neg_m, self.m_n1_1))

    def test_k(self):
        """Test identity:
        K(m) = R_F(0, 1-m, 1)
        """
        m = self.ms_up_to_1
        assert_allclose(ellipk(m), elliprf(0.0, 1.0 - m, 1.0))

    def test_km1(self):
        """Test identity:
        K(m) = R_F(0, 1-m, 1)
        But with the ellipkm1 function
        """
        tiny = finfo(double).tiny
        m1 = tiny * 2.0 ** arange(0.0, -np.log2(tiny))
        assert_allclose(ellipkm1(m1), elliprf(0.0, m1, 1.0))

    def test_e(self):
        """Test identity:
        E(m) = 2*R_G(0, 1-k^2, 1)
        """
        m = self.ms_up_to_1
        assert_allclose(ellipe(m), 2.0 * elliprg(0.0, 1.0 - m, 1.0))