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
def _ode_lie_group_try_heuristic(eq, heuristic, func, match, inf):
    xi = Function('xi')
    eta = Function('eta')
    f = func.func
    x = func.args[0]
    y = match['y']
    h = match['h']
    tempsol = []
    if not inf:
        try:
            inf = infinitesimals(eq, hint=heuristic, func=func, order=1, match=match)
        except ValueError:
            return None
    for infsim in inf:
        xiinf = infsim[xi(x, func)].subs(func, y)
        etainf = infsim[eta(x, func)].subs(func, y)
        if simplify(etainf / xiinf) == h:
            continue
        rpde = f(x, y).diff(x) * xiinf + f(x, y).diff(y) * etainf
        r = pdsolve(rpde, func=f(x, y)).rhs
        s = pdsolve(rpde - 1, func=f(x, y)).rhs
        newcoord = [_lie_group_remove(coord) for coord in [r, s]]
        r = Dummy('r')
        s = Dummy('s')
        C1 = Symbol('C1')
        rcoord = newcoord[0]
        scoord = newcoord[-1]
        try:
            sol = solve([r - rcoord, s - scoord], x, y, dict=True)
            if sol == []:
                continue
        except NotImplementedError:
            continue
        else:
            sol = sol[0]
            xsub = sol[x]
            ysub = sol[y]
            num = simplify(scoord.diff(x) + scoord.diff(y) * h)
            denom = simplify(rcoord.diff(x) + rcoord.diff(y) * h)
            if num and denom:
                diffeq = simplify((num / denom).subs([(x, xsub), (y, ysub)]))
                sep = separatevars(diffeq, symbols=[r, s], dict=True)
                if sep:
                    deq = integrate(1 / sep[s], s) + C1 - integrate(sep['coeff'] * sep[r], r)
                    deq = deq.subs([(r, rcoord), (s, scoord)])
                    try:
                        sdeq = solve(deq, y)
                    except NotImplementedError:
                        tempsol.append(deq)
                    else:
                        return [Eq(f(x), sol) for sol in sdeq]
            elif denom:
                return [Eq(f(x), solve(scoord - C1, y)[0])]
            elif num:
                return [Eq(f(x), solve(rcoord - C1, y)[0])]
    if tempsol:
        return [Eq(sol.subs(y, f(x)), 0) for sol in tempsol]
    return None