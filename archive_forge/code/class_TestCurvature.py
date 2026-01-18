import unittest
from cvxpy import Constant, Variable, log
from cvxpy.settings import QUASILINEAR, UNKNOWN
class TestCurvature(unittest.TestCase):
    """ Unit tests for the expression/curvature class. """

    def setUp(self) -> None:
        self.cvx = Variable() ** 2
        self.ccv = Variable() ** 0.5
        self.aff = Variable()
        self.const = Constant(5)
        self.unknown_curv = log(Variable() ** 3)
        self.pos = Constant(1)
        self.neg = Constant(-1)
        self.zero = Constant(0)
        self.unknown_sign = self.pos + self.neg

    def test_add(self) -> None:
        self.assertEqual((self.const + self.cvx).curvature, self.cvx.curvature)
        self.assertEqual((self.unknown_curv + self.ccv).curvature, UNKNOWN)
        self.assertEqual((self.cvx + self.ccv).curvature, UNKNOWN)
        self.assertEqual((self.cvx + self.cvx).curvature, self.cvx.curvature)
        self.assertEqual((self.aff + self.ccv).curvature, self.ccv.curvature)

    def test_sub(self) -> None:
        self.assertEqual((self.const - self.cvx).curvature, self.ccv.curvature)
        self.assertEqual((self.unknown_curv - self.ccv).curvature, UNKNOWN)
        self.assertEqual((self.cvx - self.ccv).curvature, self.cvx.curvature)
        self.assertEqual((self.cvx - self.cvx).curvature, UNKNOWN)
        self.assertEqual((self.aff - self.ccv).curvature, self.cvx.curvature)

    def test_sign_mult(self) -> None:
        self.assertEqual((self.zero * self.cvx).curvature, self.aff.curvature)
        self.assertEqual((self.neg * self.cvx).curvature, self.ccv.curvature)
        self.assertEqual((self.neg * self.ccv).curvature, self.cvx.curvature)
        self.assertEqual((self.neg * self.unknown_curv).curvature, QUASILINEAR)
        self.assertEqual((self.pos * self.aff).curvature, self.aff.curvature)
        self.assertEqual((self.pos * self.ccv).curvature, self.ccv.curvature)
        self.assertEqual((self.unknown_sign * self.const).curvature, self.const.curvature)
        self.assertEqual((self.unknown_sign * self.ccv).curvature, UNKNOWN)

    def test_neg(self) -> None:
        self.assertEqual((-self.cvx).curvature, self.ccv.curvature)
        self.assertEqual((-self.aff).curvature, self.aff.curvature)

    def test_is_curvature(self) -> None:
        assert self.const.is_affine()
        assert self.aff.is_affine()
        assert not self.cvx.is_affine()
        assert not self.ccv.is_affine()
        assert not self.unknown_curv.is_affine()
        assert self.const.is_convex()
        assert self.aff.is_convex()
        assert self.cvx.is_convex()
        assert not self.ccv.is_convex()
        assert not self.unknown_curv.is_convex()
        assert self.const.is_concave()
        assert self.aff.is_concave()
        assert not self.cvx.is_concave()
        assert self.ccv.is_concave()
        assert not self.unknown_curv.is_concave()