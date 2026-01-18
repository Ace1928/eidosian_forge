from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
class TestPolySubclassing:

    class P(PPoly):
        pass

    class B(BPoly):
        pass

    def _make_polynomials(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(3))
        c = np.random.random((4, 2))
        return (self.P(c, x), self.B(c, x))

    def test_derivative(self):
        pp, bp = self._make_polynomials()
        for p in (pp, bp):
            pd = p.derivative()
            assert_equal(p.__class__, pd.__class__)
        ppa = pp.antiderivative()
        assert_equal(pp.__class__, ppa.__class__)

    def test_from_spline(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0)
        pp = self.P.from_spline(spl)
        assert_equal(pp.__class__, self.P)

    def test_conversions(self):
        pp, bp = self._make_polynomials()
        pp1 = self.P.from_bernstein_basis(bp)
        assert_equal(pp1.__class__, self.P)
        bp1 = self.B.from_power_basis(pp)
        assert_equal(bp1.__class__, self.B)

    def test_from_derivatives(self):
        x = [0, 1, 2]
        y = [[1], [2], [3]]
        bp = self.B.from_derivatives(x, y)
        assert_equal(bp.__class__, self.B)