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
def _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, b=None, P=None):
    if J is None or f_t is None or (not _matrix_is_constant(J, t)):
        raise ValueError(filldedent("\n            Correctly input for args 'A' and 'f_t' for Linear, Higher Order,\n            Type 2\n        "))
    if P is None and b is not None and (not b.is_zero_matrix):
        raise ValueError(filldedent("\n            Provide the keyword 'P' for matrix P in A = P * J * P-1.\n        "))
    new_funcs = Matrix([Function(Dummy('{}__0'.format(f.func.__name__)))(t) for f in funcs])
    new_eqs = new_funcs.diff(t, max_order) - f_t * J * new_funcs
    if b is not None and (not b.is_zero_matrix):
        new_eqs -= P.inv() * b
    new_eqs = canonical_odes(new_eqs, new_funcs, t)[0]
    return (new_eqs, new_funcs)