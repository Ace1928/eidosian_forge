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
def _helper_simplify(eq, hint, match, simplify=True, ics=None, **kwargs):
    """
    Helper function of dsolve that calls the respective
    :py:mod:`~sympy.solvers.ode` functions to solve for the ordinary
    differential equations. This minimizes the computation in calling
    :py:meth:`~sympy.solvers.deutils._desolve` multiple times.
    """
    r = match
    func = r['func']
    order = r['order']
    match = r[hint]
    if isinstance(match, SingleODESolver):
        solvefunc = match
    elif hint.endswith('_Integral'):
        solvefunc = globals()['ode_' + hint[:-len('_Integral')]]
    else:
        solvefunc = globals()['ode_' + hint]
    free = eq.free_symbols
    cons = lambda s: s.free_symbols.difference(free)
    if simplify:
        if isinstance(solvefunc, SingleODESolver):
            sols = solvefunc.get_general_solution()
        else:
            sols = solvefunc(eq, func, order, match)
        if iterable(sols):
            rv = [odesimp(eq, s, func, hint) for s in sols]
        else:
            rv = odesimp(eq, sols, func, hint)
    else:
        if isinstance(solvefunc, SingleODESolver):
            exprs = solvefunc.get_general_solution(simplify=False)
        else:
            match['simplify'] = False
            exprs = solvefunc(eq, func, order, match)
        if isinstance(exprs, list):
            rv = [_handle_Integral(expr, func, hint) for expr in exprs]
        else:
            rv = _handle_Integral(exprs, func, hint)
    if isinstance(rv, list):
        if simplify:
            rv = _remove_redundant_solutions(eq, rv, order, func.args[0])
        if len(rv) == 1:
            rv = rv[0]
    if ics and 'power_series' not in hint:
        if isinstance(rv, (Expr, Eq)):
            solved_constants = solve_ics([rv], [r['func']], cons(rv), ics)
            rv = rv.subs(solved_constants)
        else:
            rv1 = []
            for s in rv:
                try:
                    solved_constants = solve_ics([s], [r['func']], cons(s), ics)
                except ValueError:
                    continue
                rv1.append(s.subs(solved_constants))
            if len(rv1) == 1:
                return rv1[0]
            rv = rv1
    return rv