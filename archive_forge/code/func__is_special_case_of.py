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
def _is_special_case_of(soln1, soln2, eq, order, var):
    """
    True if soln1 is found to be a special case of soln2 wrt some value of the
    constants that appear in soln2. False otherwise.
    """
    soln1 = soln1.rhs - soln1.lhs
    soln2 = soln2.rhs - soln2.lhs
    if soln1.has(Order) and soln2.has(Order):
        if soln1.getO() == soln2.getO():
            soln1 = soln1.removeO()
            soln2 = soln2.removeO()
        else:
            return False
    elif soln1.has(Order) or soln2.has(Order):
        return False
    constants1 = soln1.free_symbols.difference(eq.free_symbols)
    constants2 = soln2.free_symbols.difference(eq.free_symbols)
    constants1_new = get_numbered_constants(Tuple(soln1, soln2), len(constants1))
    if len(constants1) == 1:
        constants1_new = {constants1_new}
    for c_old, c_new in zip(constants1, constants1_new):
        soln1 = soln1.subs(c_old, c_new)
    lhs = soln1
    rhs = soln2
    eqns = [Eq(lhs, rhs)]
    for n in range(1, order):
        lhs = lhs.diff(var)
        rhs = rhs.diff(var)
        eq = Eq(lhs, rhs)
        eqns.append(eq)
    if any((isinstance(eq, BooleanFalse) for eq in eqns)):
        return False
    eqns = [eq for eq in eqns if not isinstance(eq, BooleanTrue)]
    try:
        constant_solns = solve(eqns, constants2)
    except NotImplementedError:
        return False
    if isinstance(constant_solns, dict):
        constant_solns = [constant_solns]
    for constant_soln in constant_solns:
        for eq in eqns:
            eq = eq.rhs - eq.lhs
            if checksol(eq, constant_soln) is not True:
                return False
    for constant_soln in constant_solns:
        if not any((c.has(var) for c in constant_soln.values())):
            return True
    return False