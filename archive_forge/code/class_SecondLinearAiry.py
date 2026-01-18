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
class SecondLinearAiry(SingleODESolver):
    """
    Gives solution of the Airy differential equation

    .. math :: \\frac{d^2y}{dx^2} + (a + b x) y(x) = 0

    in terms of Airy special functions airyai and airybi.

    Examples
    ========

    >>> from sympy import dsolve, Function
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = f(x).diff(x, 2) - x*f(x)
    >>> dsolve(eq)
    Eq(f(x), C1*airyai(x) + C2*airybi(x))
    """
    hint = '2nd_linear_airy'
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a4 = Wild('a4', exclude=[x, f, df])
        b4 = Wild('b4', exclude=[x, f, df])
        match = self.ode_problem.get_linear_coefficients(eq, f, order)
        does_match = False
        if order == 2 and match and (match[2] != 0):
            if match[1].is_zero:
                self.rn = cancel(match[0] / match[2]).match(a4 + b4 * x)
                if self.rn and self.rn[b4] != 0:
                    self.rn = {'b': self.rn[a4], 'm': self.rn[b4]}
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool=True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        b = self.rn['b']
        m = self.rn['m']
        if m.is_positive:
            arg = -b / cbrt(m) ** 2 - cbrt(m) * x
        elif m.is_negative:
            arg = -b / cbrt(-m) ** 2 + cbrt(-m) * x
        else:
            arg = -b / cbrt(-m) ** 2 + cbrt(-m) * x
        return [Eq(f(x), C1 * airyai(arg) + C2 * airybi(arg))]