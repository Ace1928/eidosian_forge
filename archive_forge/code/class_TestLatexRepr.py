from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
class TestLatexRepr:
    """Test the latex repr used by Jupyter"""

    def as_latex(self, obj):
        obj._repr_latex_scalar = lambda x, parens=False: str(x)
        try:
            return obj._repr_latex_()
        finally:
            del obj._repr_latex_scalar

    def test_simple_polynomial(self):
        p = poly.Polynomial([1, 2, 3])
        assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,x + 3.0\\,x^{2}$')
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0])
        assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,\\left(1.0 + x\\right) + 3.0\\,\\left(1.0 + x\\right)^{2}$')
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5])
        assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,\\left(2.0x\\right) + 3.0\\,\\left(2.0x\\right)^{2}$')
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p), '$x \\mapsto 1.0 + 2.0\\,\\left(1.0 + 2.0x\\right) + 3.0\\,\\left(1.0 + 2.0x\\right)^{2}$')

    def test_basis_func(self):
        p = poly.Chebyshev([1, 2, 3])
        assert_equal(self.as_latex(p), '$x \\mapsto 1.0\\,{T}_{0}(x) + 2.0\\,{T}_{1}(x) + 3.0\\,{T}_{2}(x)$')
        p = poly.Chebyshev([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p), '$x \\mapsto 1.0\\,{T}_{0}(1.0 + 2.0x) + 2.0\\,{T}_{1}(1.0 + 2.0x) + 3.0\\,{T}_{2}(1.0 + 2.0x)$')

    def test_multichar_basis_func(self):
        p = poly.HermiteE([1, 2, 3])
        assert_equal(self.as_latex(p), '$x \\mapsto 1.0\\,{He}_{0}(x) + 2.0\\,{He}_{1}(x) + 3.0\\,{He}_{2}(x)$')

    def test_symbol_basic(self):
        p = poly.Polynomial([1, 2, 3], symbol='z')
        assert_equal(self.as_latex(p), '$z \\mapsto 1.0 + 2.0\\,z + 3.0\\,z^{2}$')
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0], symbol='z')
        assert_equal(self.as_latex(p), '$z \\mapsto 1.0 + 2.0\\,\\left(1.0 + z\\right) + 3.0\\,\\left(1.0 + z\\right)^{2}$')
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5], symbol='z')
        assert_equal(self.as_latex(p), '$z \\mapsto 1.0 + 2.0\\,\\left(2.0z\\right) + 3.0\\,\\left(2.0z\\right)^{2}$')
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0], symbol='z')
        assert_equal(self.as_latex(p), '$z \\mapsto 1.0 + 2.0\\,\\left(1.0 + 2.0z\\right) + 3.0\\,\\left(1.0 + 2.0z\\right)^{2}$')