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
class NthLinearConstantCoeffUndeterminedCoefficients(SingleODESolver):
    """
    Solves an `n`\\th order linear differential equation with constant
    coefficients using the method of undetermined coefficients.

    This method works on differential equations of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \\cdots + a_1 f'(x)
                + a_0 f(x) = P(x)\\text{,}

    where `P(x)` is a function that has a finite number of linearly
    independent derivatives.

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \\sin(c x + d)` or `a x^i e^{b x} \\cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \\sin(x)`,
    and `e^x \\cos(x)` can all be used.  Products of `\\sin`'s and `\\cos`'s have
    a finite number of derivatives, because they can be expanded into `\\sin(a
    x)` and `\\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\\sin^2(x)` into `(1 + \\cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.

    This method works by creating a trial function from the expression and all
    of its linear independent derivatives and substituting them into the
    original ODE.  The coefficients for each term will be a system of linear
    equations, which are be solved for and substituted, giving the solution.
    If any of the trial functions are linearly dependent on the solution to
    the homogeneous equation, they are multiplied by sufficient `x` to make
    them linearly independent.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 2) + 2*f(x).diff(x) + f(x) -
    ... 4*exp(-x)*x**2 + cos(2*x), f(x),
    ... hint='nth_linear_constant_coeff_undetermined_coefficients'))
           /       /      3\\\\
           |       |     x ||  -x   4*sin(2*x)   3*cos(2*x)
    f(x) = |C1 + x*|C2 + --||*e   - ---------- + ----------
           \\       \\     3 //           25           25

    References
    ==========

    - https://en.wikipedia.org/wiki/Method_of_undetermined_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 221

    # indirect doctest

    """
    hint = 'nth_linear_constant_coeff_undetermined_coefficients'
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        does_match = False
        if order and self.r and (not any((self.r[i].has(x) for i in self.r if i >= 0))):
            if self.r[-1]:
                eq_homogeneous = Add(eq, -self.r[-1])
                undetcoeff = _undetermined_coefficients_match(self.r[-1], x, func, eq_homogeneous)
                if undetcoeff['test']:
                    self.trialset = undetcoeff['trialset']
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool=True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        homogen_sol = Add(*[i * j for i, j in zip(constants, roots)])
        homogen_sol = Eq(f(x), homogen_sol)
        self.r.update({'list': roots, 'sol': homogen_sol, 'simpliy_flag': simplify_flag})
        gsol = _solve_undetermined_coefficients(eq, f(x), order, self.r, self.trialset)
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], f(x), collectterms)
        return [gsol]