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
def _nonlinear_3eq_order1_type3(x, y, z, t, eq):
    """
    Equations:

    .. math:: x' = c F_2 - b F_3, \\enspace y' = a F_3 - c F_1, \\enspace z' = b F_1 - a F_2

    where `F_n = F_n(x, y, z, t)`.

    1. First Integral:

    .. math:: a x + b y + c z = C_1,

    where C is an arbitrary constant.

    2. If we assume function `F_n` to be independent of `t`,i.e, `F_n` = `F_n (x, y, z)`
    Then, on eliminating `t` and `z` from the first two equation of the system, one
    arrives at the first-order equation

    .. math:: \\frac{dy}{dx} = \\frac{a F_3 (x, y, z) - c F_1 (x, y, z)}{c F_2 (x, y, z) -
                b F_3 (x, y, z)}

    where `z = \\frac{1}{c} (C_1 - a x - b y)`

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0404.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    fu, fv, fw = symbols('u, v, w', cls=Function)
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = (diff(x(t), t) - eq[0]).match(F2 - F3)
    r = collect_const(r1[F2]).match(s * F2)
    r.update(collect_const(r1[F3]).match(q * F3))
    if eq[1].has(r[F2]) and (not eq[1].has(r[F3])):
        r[F2], r[F3] = (r[F3], r[F2])
        r[s], r[q] = (-r[q], -r[s])
    r.update((diff(y(t), t) - eq[1]).match(p * r[F3] - r[s] * F1))
    a = r[p]
    b = r[q]
    c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    z_xy = (C1 - a * u - b * v) / c
    y_zx = (C1 - a * u - c * w) / b
    x_yz = (C1 - b * v - c * w) / a
    y_x = dsolve(diff(fv(u), u) - ((a * F3 - c * F1) / (c * F2 - b * F3)).subs(w, z_xy).subs(v, fv(u))).rhs
    z_x = dsolve(diff(fw(u), u) - ((b * F1 - a * F2) / (c * F2 - b * F3)).subs(v, y_zx).subs(w, fw(u))).rhs
    z_y = dsolve(diff(fw(v), v) - ((b * F1 - a * F2) / (a * F3 - c * F1)).subs(u, x_yz).subs(w, fw(v))).rhs
    x_y = dsolve(diff(fu(v), v) - ((c * F2 - b * F3) / (a * F3 - c * F1)).subs(w, z_xy).subs(u, fu(v))).rhs
    y_z = dsolve(diff(fv(w), w) - ((a * F3 - c * F1) / (b * F1 - a * F2)).subs(u, x_yz).subs(v, fv(w))).rhs
    x_z = dsolve(diff(fu(w), w) - ((c * F2 - b * F3) / (b * F1 - a * F2)).subs(v, y_zx).subs(u, fu(w))).rhs
    sol1 = dsolve(diff(fu(t), t) - (c * F2 - b * F3).subs(v, y_x).subs(w, z_x).subs(u, fu(t))).rhs
    sol2 = dsolve(diff(fv(t), t) - (a * F3 - c * F1).subs(u, x_y).subs(w, z_y).subs(v, fv(t))).rhs
    sol3 = dsolve(diff(fw(t), t) - (b * F1 - a * F2).subs(u, x_z).subs(v, y_z).subs(w, fw(t))).rhs
    return [sol1, sol2, sol3]