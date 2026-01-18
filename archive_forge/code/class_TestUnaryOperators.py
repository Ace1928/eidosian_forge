import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
class TestUnaryOperators:
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_neg(self):
        n = -self.p
        assert_equal(n.symbol, 'z')

    def test_scalarmul(self):
        out = self.p * 10
        assert_equal(out.symbol, 'z')

    def test_rscalarmul(self):
        out = 10 * self.p
        assert_equal(out.symbol, 'z')

    def test_pow(self):
        out = self.p ** 3
        assert_equal(out.symbol, 'z')