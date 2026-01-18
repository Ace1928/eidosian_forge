from itertools import islice
from sympy.core import Add, S, Mul, Pow
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.functions import exp, log
from sympy.integrals.integrals import integrate
from sympy.polys import Poly
from sympy.polys.polytools import cancel, div
from sympy.simplify import (collect, powsimp,  # type: ignore
from sympy.solvers import solve
from sympy.solvers.pde import pdsolve
from sympy.utilities import numbered_symbols
from sympy.solvers.deutils import _preprocess, ode_order
from .ode import checkinfsol
def _ode_lie_group(s, func, order, match):
    heuristics = lie_heuristics
    inf = {}
    f = func.func
    x = func.args[0]
    df = func.diff(x)
    xi = Function('xi')
    eta = Function('eta')
    xis = match['xi']
    etas = match['eta']
    y = match.pop('y', None)
    if y:
        h = -simplify(match[match['d']] / match[match['e']])
        y = y
    else:
        y = Dummy('y')
        h = s.subs(func, y)
    if xis is not None and etas is not None:
        inf = [{xi(x, f(x)): S(xis), eta(x, f(x)): S(etas)}]
        if checkinfsol(Eq(df, s), inf, func=f(x), order=1)[0][0]:
            heuristics = ['user_defined'] + list(heuristics)
    match = {'h': h, 'y': y}
    sol = None
    for heuristic in heuristics:
        sol = _ode_lie_group_try_heuristic(Eq(df, s), heuristic, func, match, inf)
        if sol:
            return sol
    return sol