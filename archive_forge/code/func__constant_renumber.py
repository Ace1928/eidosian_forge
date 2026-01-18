from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve
from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from .single import SingleODEProblem, SingleODESolver, solver_map
def _constant_renumber(expr):
    """
        We need to have an internal recursive function
        """
    if isinstance(expr, Tuple):
        renumbered = [_constant_renumber(e) for e in expr]
        return Tuple(*renumbered)
    if isinstance(expr, Equality):
        return Eq(_constant_renumber(expr.lhs), _constant_renumber(expr.rhs))
    if type(expr) not in (Mul, Add, Pow) and (not expr.is_Function) and (not expr.has(*constantsymbols)):
        return expr
    elif expr.is_Piecewise:
        return expr
    elif expr in constantsymbols:
        if expr not in constants_found:
            constants_found.append(expr)
        return expr
    elif expr.is_Function or expr.is_Pow:
        return expr.func(*[_constant_renumber(x) for x in expr.args])
    else:
        sortedargs = list(expr.args)
        sortedargs.sort(key=sort_key)
        return expr.func(*[_constant_renumber(x) for x in sortedargs])