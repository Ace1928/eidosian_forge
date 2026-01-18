from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
class TestStrUnicodeSuperSubscripts:

    @pytest.fixture(scope='class', autouse=True)
    def use_unicode(self):
        poly.set_default_printstyle('unicode')

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0·x + 3.0·x²'), ([-1, 0, 3, -1], '-1.0 + 0.0·x + 3.0·x² - 1.0·x³'), (arange(12), '0.0 + 1.0·x + 2.0·x² + 3.0·x³ + 4.0·x⁴ + 5.0·x⁵ + 6.0·x⁶ + 7.0·x⁷ +\n8.0·x⁸ + 9.0·x⁹ + 10.0·x¹⁰ + 11.0·x¹¹')))
    def test_polynomial_str(self, inp, tgt):
        res = str(poly.Polynomial(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0·T₁(x) + 3.0·T₂(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0·T₁(x) + 3.0·T₂(x) - 1.0·T₃(x)'), (arange(12), '0.0 + 1.0·T₁(x) + 2.0·T₂(x) + 3.0·T₃(x) + 4.0·T₄(x) + 5.0·T₅(x) +\n6.0·T₆(x) + 7.0·T₇(x) + 8.0·T₈(x) + 9.0·T₉(x) + 10.0·T₁₀(x) + 11.0·T₁₁(x)')))
    def test_chebyshev_str(self, inp, tgt):
        res = str(poly.Chebyshev(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0·P₁(x) + 3.0·P₂(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0·P₁(x) + 3.0·P₂(x) - 1.0·P₃(x)'), (arange(12), '0.0 + 1.0·P₁(x) + 2.0·P₂(x) + 3.0·P₃(x) + 4.0·P₄(x) + 5.0·P₅(x) +\n6.0·P₆(x) + 7.0·P₇(x) + 8.0·P₈(x) + 9.0·P₉(x) + 10.0·P₁₀(x) + 11.0·P₁₁(x)')))
    def test_legendre_str(self, inp, tgt):
        res = str(poly.Legendre(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0·H₁(x) + 3.0·H₂(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0·H₁(x) + 3.0·H₂(x) - 1.0·H₃(x)'), (arange(12), '0.0 + 1.0·H₁(x) + 2.0·H₂(x) + 3.0·H₃(x) + 4.0·H₄(x) + 5.0·H₅(x) +\n6.0·H₆(x) + 7.0·H₇(x) + 8.0·H₈(x) + 9.0·H₉(x) + 10.0·H₁₀(x) + 11.0·H₁₁(x)')))
    def test_hermite_str(self, inp, tgt):
        res = str(poly.Hermite(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0·He₁(x) + 3.0·He₂(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0·He₁(x) + 3.0·He₂(x) - 1.0·He₃(x)'), (arange(12), '0.0 + 1.0·He₁(x) + 2.0·He₂(x) + 3.0·He₃(x) + 4.0·He₄(x) + 5.0·He₅(x) +\n6.0·He₆(x) + 7.0·He₇(x) + 8.0·He₈(x) + 9.0·He₉(x) + 10.0·He₁₀(x) +\n11.0·He₁₁(x)')))
    def test_hermiteE_str(self, inp, tgt):
        res = str(poly.HermiteE(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0·L₁(x) + 3.0·L₂(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0·L₁(x) + 3.0·L₂(x) - 1.0·L₃(x)'), (arange(12), '0.0 + 1.0·L₁(x) + 2.0·L₂(x) + 3.0·L₃(x) + 4.0·L₄(x) + 5.0·L₅(x) +\n6.0·L₆(x) + 7.0·L₇(x) + 8.0·L₈(x) + 9.0·L₉(x) + 10.0·L₁₀(x) + 11.0·L₁₁(x)')))
    def test_laguerre_str(self, inp, tgt):
        res = str(poly.Laguerre(inp))
        assert_equal(res, tgt)