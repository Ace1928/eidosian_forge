from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate
def _is_second_order_type2(A, t):
    term = _factor_matrix(A, t)
    is_type2 = False
    if term is not None:
        term = 1 / term[0]
        is_type2 = term.is_polynomial()
    if is_type2:
        poly = Poly(term.expand(), t)
        monoms = poly.monoms()
        if monoms[0][0] in (2, 4):
            cs = _get_poly_coeffs(poly, 4)
            a, b, c, d, e = cs
            a1 = powdenest(sqrt(a), force=True)
            c1 = powdenest(sqrt(e), force=True)
            b1 = powdenest(sqrt(c - 2 * a1 * c1), force=True)
            is_type2 = b == 2 * a1 * b1 and d == 2 * b1 * c1
            term = a1 * t ** 2 + b1 * t + c1
        else:
            is_type2 = False
    return (is_type2, term)