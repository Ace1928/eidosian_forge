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
class SecondNonlinearAutonomousConserved(SinglePatternODESolver):
    """
    Gives solution for the autonomous second order nonlinear
    differential equation of the form

    .. math :: f''(x) = g(f(x))

    The solution for this differential equation can be computed
    by multiplying by `f'(x)` and integrating on both sides,
    converting it into a first order differential equation.

    Examples
    ========

    >>> from sympy import Function, symbols, dsolve
    >>> f, g = symbols('f g', cls=Function)
    >>> x = symbols('x')

    >>> eq = f(x).diff(x, 2) - g(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 - x)]

    >>> from sympy import exp, log
    >>> eq = f(x).diff(x, 2) - exp(f(x)) + log(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 - x)]

    References
    ==========

    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0301.pdf
    """
    hint = '2nd_nonlinear_autonomous_conserved'
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        fy = Wild('fy', exclude=[0, f(x).diff(x), f(x).diff(x, 2)])
        return (fy,)

    def _equation(self, fx, x, order):
        fy = self.wilds()[0]
        return fx.diff(x, 2) + fy

    def _verify(self, fx):
        return self.ode_problem.is_autonomous

    def _get_general_solution(self, *, simplify_flag: bool=True):
        g = self.wilds_match()[0]
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = Dummy('u')
        g = g.subs(fx, u)
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        inside = -2 * Integral(g, u) + C1
        lhs = Integral(1 / sqrt(inside), (u, fx))
        return [Eq(lhs, C2 + x), Eq(lhs, C2 - x)]