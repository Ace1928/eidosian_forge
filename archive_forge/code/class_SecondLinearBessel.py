from __future__ import annotations
from typing import ClassVar, Iterator
from .riccati import match_riccati, solve_riccati
from sympy.core import Add, S, Pow, Rational
from sympy.core.cache import cached_property
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, Function, expand, Subs, _mexpand
from sympy.core.numbers import zoo
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Dummy, Wild
from sympy.core.mul import Mul
from sympy.functions import exp, tan, log, sqrt, besselj, bessely, cbrt, airyai, airybi
from sympy.integrals import Integral
from sympy.polys import Poly
from sympy.polys.polytools import cancel, factor, degree
from sympy.simplify import collect, simplify, separatevars, logcombine, posify # type: ignore
from sympy.simplify.radsimp import fraction
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.solvers.deutils import ode_order, _preprocess
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.solvers import PolyNonlinearError
from .hypergeometric import equivalence_hypergeometric, match_2nd_2F1_hypergeometric, \
from .nonhomogeneous import _get_euler_characteristic_eq_sols, _get_const_characteristic_eq_sols, \
from .lie_group import _ode_lie_group
from .ode import dsolve, ode_sol_simplicity, odesimp, homogeneous_order
class SecondLinearBessel(SingleODESolver):
    """
    Gives solution of the Bessel differential equation

    .. math :: x^2 \\frac{d^2y}{dx^2} + x \\frac{dy}{dx} y(x) + (x^2-n^2) y(x)

    if `n` is integer then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))`` as both the solutions are linearly independent else if
    `n` is a fraction then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 besselj(-n,x))`` which can also transform into ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))``.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import Symbol
    >>> v = Symbol('v', positive=True)
    >>> from sympy import dsolve, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - v**2)*y
    >>> dsolve(genform)
    Eq(f(x), C1*besselj(v, x) + C2*bessely(v, x))

    References
    ==========

    https://math24.net/bessel-differential-equation.html

    """
    hint = '2nd_linear_bessel'
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a = Wild('a', exclude=[f, df])
        b = Wild('b', exclude=[x, f, df])
        a4 = Wild('a4', exclude=[x, f, df])
        b4 = Wild('b4', exclude=[x, f, df])
        c4 = Wild('c4', exclude=[x, f, df])
        d4 = Wild('d4', exclude=[x, f, df])
        a3 = Wild('a3', exclude=[f, df, f.diff(x, 2)])
        b3 = Wild('b3', exclude=[f, df, f.diff(x, 2)])
        c3 = Wild('c3', exclude=[f, df, f.diff(x, 2)])
        deq = a3 * f.diff(x, 2) + b3 * df + c3 * f
        r = collect(eq, [f.diff(x, 2), df, f]).match(deq)
        if order == 2 and r:
            if not all((r[key].is_polynomial() for key in r)):
                n, d = eq.as_numer_denom()
                eq = expand(n)
                r = collect(eq, [f.diff(x, 2), df, f]).match(deq)
        if r and r[a3] != 0:
            coeff = factor(r[a3]).match(a4 * (x - b) ** b4)
            if coeff:
                if coeff[b4] == 0:
                    return False
                point = coeff[b]
            else:
                return False
            if point:
                r[a3] = simplify(r[a3].subs(x, x + point))
                r[b3] = simplify(r[b3].subs(x, x + point))
                r[c3] = simplify(r[c3].subs(x, x + point))
            r[a3] = cancel(r[a3] / (coeff[a4] * x ** (-2 + coeff[b4])))
            r[b3] = cancel(r[b3] / (coeff[a4] * x ** (-2 + coeff[b4])))
            r[c3] = cancel(r[c3] / (coeff[a4] * x ** (-2 + coeff[b4])))
            coeff1 = factor(r[b3]).match(a4 * x)
            if coeff1 is None:
                return False
            _coeff2 = r[c3].match(a - b)
            if _coeff2 is None:
                return False
            coeff2 = factor(_coeff2[a]).match(c4 ** 2 * x ** (2 * a4))
            if coeff2 is None:
                return False
            if _coeff2[b] == 0:
                coeff2[d4] = 0
            else:
                coeff2[d4] = factor(_coeff2[b]).match(d4 ** 2)[d4]
            self.rn = {'n': coeff2[d4], 'a4': coeff2[c4], 'd4': coeff2[a4]}
            self.rn['c4'] = coeff1[a4]
            self.rn['b4'] = point
            return True
        return False

    def _get_general_solution(self, *, simplify_flag: bool=True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.rn['n']
        a4 = self.rn['a4']
        c4 = self.rn['c4']
        d4 = self.rn['d4']
        b4 = self.rn['b4']
        n = sqrt(n ** 2 + Rational(1, 4) * (c4 - 1) ** 2)
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        return [Eq(f(x), (x ** Rational(1 - c4, 2) * (C1 * besselj(n / d4, a4 * x ** d4 / d4) + C2 * bessely(n / d4, a4 * x ** d4 / d4))).subs(x, x - b4))]