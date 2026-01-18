from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
class TestStrAscii:

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 x + 3.0 x**2'), ([-1, 0, 3, -1], '-1.0 + 0.0 x + 3.0 x**2 - 1.0 x**3'), (arange(12), '0.0 + 1.0 x + 2.0 x**2 + 3.0 x**3 + 4.0 x**4 + 5.0 x**5 + 6.0 x**6 +\n7.0 x**7 + 8.0 x**8 + 9.0 x**9 + 10.0 x**10 + 11.0 x**11')))
    def test_polynomial_str(self, inp, tgt):
        res = str(poly.Polynomial(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 T_1(x) + 3.0 T_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 T_1(x) + 3.0 T_2(x) - 1.0 T_3(x)'), (arange(12), '0.0 + 1.0 T_1(x) + 2.0 T_2(x) + 3.0 T_3(x) + 4.0 T_4(x) + 5.0 T_5(x) +\n6.0 T_6(x) + 7.0 T_7(x) + 8.0 T_8(x) + 9.0 T_9(x) + 10.0 T_10(x) +\n11.0 T_11(x)')))
    def test_chebyshev_str(self, inp, tgt):
        res = str(poly.Chebyshev(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 P_1(x) + 3.0 P_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 P_1(x) + 3.0 P_2(x) - 1.0 P_3(x)'), (arange(12), '0.0 + 1.0 P_1(x) + 2.0 P_2(x) + 3.0 P_3(x) + 4.0 P_4(x) + 5.0 P_5(x) +\n6.0 P_6(x) + 7.0 P_7(x) + 8.0 P_8(x) + 9.0 P_9(x) + 10.0 P_10(x) +\n11.0 P_11(x)')))
    def test_legendre_str(self, inp, tgt):
        res = str(poly.Legendre(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 H_1(x) + 3.0 H_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 H_1(x) + 3.0 H_2(x) - 1.0 H_3(x)'), (arange(12), '0.0 + 1.0 H_1(x) + 2.0 H_2(x) + 3.0 H_3(x) + 4.0 H_4(x) + 5.0 H_5(x) +\n6.0 H_6(x) + 7.0 H_7(x) + 8.0 H_8(x) + 9.0 H_9(x) + 10.0 H_10(x) +\n11.0 H_11(x)')))
    def test_hermite_str(self, inp, tgt):
        res = str(poly.Hermite(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 He_1(x) + 3.0 He_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 He_1(x) + 3.0 He_2(x) - 1.0 He_3(x)'), (arange(12), '0.0 + 1.0 He_1(x) + 2.0 He_2(x) + 3.0 He_3(x) + 4.0 He_4(x) +\n5.0 He_5(x) + 6.0 He_6(x) + 7.0 He_7(x) + 8.0 He_8(x) + 9.0 He_9(x) +\n10.0 He_10(x) + 11.0 He_11(x)')))
    def test_hermiteE_str(self, inp, tgt):
        res = str(poly.HermiteE(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 L_1(x) + 3.0 L_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 L_1(x) + 3.0 L_2(x) - 1.0 L_3(x)'), (arange(12), '0.0 + 1.0 L_1(x) + 2.0 L_2(x) + 3.0 L_3(x) + 4.0 L_4(x) + 5.0 L_5(x) +\n6.0 L_6(x) + 7.0 L_7(x) + 8.0 L_8(x) + 9.0 L_9(x) + 10.0 L_10(x) +\n11.0 L_11(x)')))
    def test_laguerre_str(self, inp, tgt):
        res = str(poly.Laguerre(inp))
        assert_equal(res, tgt)