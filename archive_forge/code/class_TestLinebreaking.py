from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
class TestLinebreaking:

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    def test_single_line_one_less(self):
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 123])
        assert_equal(len(str(p)), 74)
        assert_equal(str(p), '12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.0 x**3 + 123.0 x**4')

    def test_num_chars_is_linewidth(self):
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 1234])
        assert_equal(len(str(p)), 75)
        assert_equal(str(p), '12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.0 x**3 +\n1234.0 x**4')

    def test_first_linebreak_multiline_one_less_than_linewidth(self):
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 1, 12345678])
        assert_equal(len(str(p).split('\n')[0]), 74)
        assert_equal(str(p), '12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.0 x**3 + 1.0 x**4 +\n12345678.0 x**5')

    def test_first_linebreak_multiline_on_linewidth(self):
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678.12, 1, 12345678])
        assert_equal(str(p), '12345678.0 + 12345678.0 x + 12345678.0 x**2 + 12345678.12 x**3 +\n1.0 x**4 + 12345678.0 x**5')

    @pytest.mark.parametrize(('lw', 'tgt'), ((75, '0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + 500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + 900.0 x**9'), (45, '0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 +\n40000.0 x**4 + 500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 +\n900.0 x**9'), (132, '0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + 500000.0 x**5 + 600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + 900.0 x**9')))
    def test_linewidth_printoption(self, lw, tgt):
        p = poly.Polynomial([0, 10, 200, 3000, 40000, 500000, 600000, 70000, 8000, 900])
        with printoptions(linewidth=lw):
            assert_equal(str(p), tgt)
            for line in str(p).split('\n'):
                assert_(len(line) < lw)