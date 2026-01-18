from __future__ import annotations
from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
from sympy.core.assumptions import check_assumptions
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand_mul, expand_log, Derivative,
from sympy.core.logic import fuzzy_not
from sympy.core.numbers import ilcm, Float, Rational, _illegal
from sympy.core.power import integer_log, Pow
from sympy.core.relational import Eq, Ne
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import And, BooleanAtom
from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.integrals import Integral
from sympy.ntheory.factor_ import divisors
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
from sympy.simplify.sqrtdenest import sqrt_depth
from sympy.simplify.fu import TR1, TR2i
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import Matrix, zeros
from sympy.polys import roots, cancel, factor, Poly
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import filldedent, debugf
from sympy.utilities.iterables import (connected_components,
from sympy.utilities.decorator import conserve_mpmath_dps
from mpmath import findroot
from sympy.solvers.polysys import solve_poly_system
from types import GeneratorType
from collections import defaultdict
from itertools import combinations, product
import warnings
from sympy.solvers.bivariate import (
def _ispow(e):
    """Return True if e is a Pow or is exp."""
    return isinstance(e, Expr) and (e.is_Pow or isinstance(e, exp))