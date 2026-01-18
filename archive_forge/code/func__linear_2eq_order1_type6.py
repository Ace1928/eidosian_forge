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
def _linear_2eq_order1_type6(x, y, t, r, eq):
    """
    The equations of this type of ode are .

    .. math:: x' = f(t) x + g(t) y

    .. math:: y' = a [f(t) + a h(t)] x + a [g(t) - h(t)] y

    This is solved by first multiplying the first equation by `-a` and adding
    it to the second equation to obtain

    .. math:: y' - a x' = -a h(t) (y - a x)

    Setting `U = y - ax` and integrating the equation we arrive at

    .. math:: y - ax = C_1 e^{-a \\int h(t) \\,dt}

    and on substituting the value of y in first equation give rise to first order ODEs. After solving for
    `x`, we can obtain `y` by substituting the value of `x` in second equation.

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    p = 0
    q = 0
    p1 = cancel(r['c'] / cancel(r['c'] / r['d']).as_numer_denom()[0])
    p2 = cancel(r['a'] / cancel(r['a'] / r['b']).as_numer_denom()[0])
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q != 0 and n == 0:
                if (r['c'] / j - r['a']) / (r['b'] - r['d'] / j) == j:
                    p = 1
                    s = j
                    break
            if q != 0 and n == 1:
                if (r['a'] / j - r['c']) / (r['d'] - r['b'] / j) == j:
                    p = 2
                    s = j
                    break
    if p == 1:
        equ = diff(x(t), t) - r['a'] * x(t) - r['b'] * (s * x(t) + C1 * exp(-s * Integral(r['b'] - r['d'] / s, t)))
        hint1 = classify_ode(equ)[1]
        sol1 = dsolve(equ, hint=hint1 + '_Integral').rhs
        sol2 = s * sol1 + C1 * exp(-s * Integral(r['b'] - r['d'] / s, t))
    elif p == 2:
        equ = diff(y(t), t) - r['c'] * y(t) - r['d'] * s * y(t) + C1 * exp(-s * Integral(r['d'] - r['b'] / s, t))
        hint1 = classify_ode(equ)[1]
        sol2 = dsolve(equ, hint=hint1 + '_Integral').rhs
        sol1 = s * sol2 + C1 * exp(-s * Integral(r['d'] - r['b'] / s, t))
    return [Eq(x(t), sol1), Eq(y(t), sol2)]