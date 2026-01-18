import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
class TestBinaryOperatorsDifferentSymbol:
    p = poly.Polynomial([1, 2, 3], symbol='x')
    other = poly.Polynomial([4, 5, 6], symbol='y')
    ops = (p.__add__, p.__sub__, p.__mul__, p.__floordiv__, p.__mod__)

    @pytest.mark.parametrize('f', ops)
    def test_binops_fails(self, f):
        assert_raises(ValueError, f, self.other)