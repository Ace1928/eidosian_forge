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
def _nonlinear_2eq_order1_type3(x, y, t, eq):
    """
    Autonomous system of general form

    .. math:: x' = F(x,y)

    .. math:: y' = G(x,y)

    Assuming `y = y(x, C_1)` where `C_1` is an arbitrary constant is the general
    solution of the first-order equation

    .. math:: F(x,y) y'_x = G(x,y)

    Then the general solution of the original system of equations has the form

    .. math:: \\int \\frac{1}{F(x,y(x,C_1))} \\,dx = t + C_1

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    v = Function('v')
    u = Symbol('u')
    f = Wild('f')
    g = Wild('g')
    r1 = eq[0].match(diff(x(t), t) - f)
    r2 = eq[1].match(diff(y(t), t) - g)
    F = r1[f].subs(x(t), u).subs(y(t), v(u))
    G = r2[g].subs(x(t), u).subs(y(t), v(u))
    sol2r = dsolve(Eq(diff(v(u), u), G / F))
    if isinstance(sol2r, Equality):
        sol2r = [sol2r]
    for sol2s in sol2r:
        sol1 = solve(Integral(1 / F.subs(v(u), sol2s.rhs), u).doit() - t - C2, u)
    sol = []
    for sols in sol1:
        sol.append(Eq(x(t), sols))
        sol.append(Eq(y(t), sol2s.rhs.subs(u, sols)))
    return sol