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
def _first_order_type5_6_subs(A, t, b=None):
    match = {}
    factor_terms = _factor_matrix(A, t)
    is_homogeneous = b is None or b.is_zero_matrix
    if factor_terms is not None:
        t_ = Symbol('{}_'.format(t))
        F_t = integrate(factor_terms[0], t)
        inverse = solveset(Eq(t_, F_t), t)
        if isinstance(inverse, FiniteSet) and (not inverse.has(Piecewise)) and (len(inverse) == 1):
            A = factor_terms[1]
            if not is_homogeneous:
                b = b / factor_terms[0]
                b = b.subs(t, list(inverse)[0])
            type = 'type{}'.format(5 + (not is_homogeneous))
            match.update({'func_coeff': A, 'tau': F_t, 't_': t_, 'type_of_equation': type, 'rhs': b})
    return match