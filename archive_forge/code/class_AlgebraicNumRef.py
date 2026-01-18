from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class AlgebraicNumRef(ArithRef):
    """Algebraic irrational values."""

    def approx(self, precision=10):
        """Return a Z3 rational number that approximates the algebraic number `self`.
        The result `r` is such that |r - self| <= 1/10^precision

        >>> x = simplify(Sqrt(2))
        >>> x.approx(20)
        6838717160008073720548335/4835703278458516698824704
        >>> x.approx(5)
        2965821/2097152
        """
        return RatNumRef(Z3_get_algebraic_number_upper(self.ctx_ref(), self.as_ast(), precision), self.ctx)

    def as_decimal(self, prec):
        """Return a string representation of the algebraic number `self` in decimal notation
        using `prec` decimal places.

        >>> x = simplify(Sqrt(2))
        >>> x.as_decimal(10)
        '1.4142135623?'
        >>> x.as_decimal(20)
        '1.41421356237309504880?'
        """
        return Z3_get_numeral_decimal_string(self.ctx_ref(), self.as_ast(), prec)

    def poly(self):
        return AstVector(Z3_algebraic_get_poly(self.ctx_ref(), self.as_ast()), self.ctx)

    def index(self):
        return Z3_algebraic_get_i(self.ctx_ref(), self.as_ast())