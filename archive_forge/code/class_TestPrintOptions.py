from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
class TestPrintOptions:
    """
    Test the output is properly configured via printoptions.
    The exponential notation is enabled automatically when the values 
    are too small or too large.
    """

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    def test_str(self):
        p = poly.Polynomial([1 / 2, 1 / 7, 1 / 7 * 10 ** 8, 1 / 7 * 10 ** 9])
        assert_equal(str(p), '0.5 + 0.14285714 x + 14285714.28571429 x**2 + (1.42857143e+08) x**3')
        with printoptions(precision=3):
            assert_equal(str(p), '0.5 + 0.143 x + 14285714.286 x**2 + (1.429e+08) x**3')

    def test_latex(self):
        p = poly.Polynomial([1 / 2, 1 / 7, 1 / 7 * 10 ** 8, 1 / 7 * 10 ** 9])
        assert_equal(p._repr_latex_(), '$x \\mapsto \\text{0.5} + \\text{0.14285714}\\,x + \\text{14285714.28571429}\\,x^{2} + \\text{(1.42857143e+08)}\\,x^{3}$')
        with printoptions(precision=3):
            assert_equal(p._repr_latex_(), '$x \\mapsto \\text{0.5} + \\text{0.143}\\,x + \\text{14285714.286}\\,x^{2} + \\text{(1.429e+08)}\\,x^{3}$')

    def test_fixed(self):
        p = poly.Polynomial([1 / 2])
        assert_equal(str(p), '0.5')
        with printoptions(floatmode='fixed'):
            assert_equal(str(p), '0.50000000')
        with printoptions(floatmode='fixed', precision=4):
            assert_equal(str(p), '0.5000')

    def test_switch_to_exp(self):
        for i, s in enumerate(SWITCH_TO_EXP):
            with printoptions(precision=i):
                p = poly.Polynomial([1.23456789 * 10 ** (-i) for i in range(i // 2 + 3)])
                assert str(p).replace('\n', ' ') == s

    def test_non_finite(self):
        p = poly.Polynomial([nan, inf])
        assert str(p) == 'nan + inf x'
        assert p._repr_latex_() == '$x \\mapsto \\text{nan} + \\text{inf}\\,x$'
        with printoptions(nanstr='NAN', infstr='INF'):
            assert str(p) == 'NAN + INF x'
            assert p._repr_latex_() == '$x \\mapsto \\text{NAN} + \\text{INF}\\,x$'