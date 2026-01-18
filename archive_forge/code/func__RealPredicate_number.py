from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow, S
from sympy.core.numbers import (AlgebraicNumber, ComplexInfinity, Exp1, Float,
from sympy.core.logic import fuzzy_bool
from sympy.functions import (Abs, acos, acot, asin, atan, cos, cot, exp, im,
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices import Determinant, MatrixBase, Trace
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.multipledispatch import MDNotImplementedError
from .common import test_closed_group
from ..predicates.sets import (IntegerPredicate, RationalPredicate,
def _RealPredicate_number(expr, assumptions):
    i = expr.as_real_imag()[1].evalf(2)
    if i._prec != 1:
        return not i