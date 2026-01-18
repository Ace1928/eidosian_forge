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
class SecondHypergeometric(SingleODESolver):
    """
    Solves 2nd order linear differential equations.

    It computes special function solutions which can be expressed using the
    2F1, 1F1 or 0F1 hypergeometric functions.

    .. math:: y'' + A(x) y' + B(x) y = 0\\text{,}

    where `A` and `B` are rational functions.

    These kinds of differential equations have solution of non-Liouvillian form.

    Given linear ODE can be obtained from 2F1 given by

    .. math:: (x^2 - x) y'' + ((a + b + 1) x - c) y' + b a y = 0\\text{,}

    where {a, b, c} are arbitrary constants.

    Notes
    =====

    The algorithm should find any solution of the form

    .. math:: y = P(x) _pF_q(..; ..;\\frac{\\alpha x^k + \\beta}{\\gamma x^k + \\delta})\\text{,}

    where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
    Currently only the 2F1 case is implemented in SymPy but the other cases are
    described in the paper and could be implemented in future (contributions
    welcome!).


    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = (x*x - x)*f(x).diff(x,2) + (5*x - 1)*f(x).diff(x) + 4*f(x)
    >>> pprint(dsolve(eq, f(x), '2nd_hypergeometric'))
                                        _
           /        /           4  \\\\  |_  /-1, -1 |  \\
           |C1 + C2*|log(x) + -----||* |   |       | x|
           \\        \\         x + 1// 2  1 \\  1    |  /
    f(x) = --------------------------------------------
                                    3
                             (x - 1)


    References
    ==========

    - "Non-Liouvillian solutions for second order linear ODEs" by L. Chan, E.S. Cheb-Terrab

    """
    hint = '2nd_hypergeometric'
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        r = match_2nd_hypergeometric(eq, func)
        self.match_object = None
        if r:
            A, B = r
            d = equivalence_hypergeometric(A, B, func)
            if d:
                if d['type'] == '2F1':
                    self.match_object = match_2nd_2F1_hypergeometric(d['I0'], d['k'], d['sing_point'], func)
                    if self.match_object is not None:
                        self.match_object.update({'A': A, 'B': B})
        return self.match_object is not None

    def _get_general_solution(self, *, simplify_flag: bool=True):
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        if self.match_object['type'] == '2F1':
            sol = get_sol_2F1_hypergeometric(eq, func, self.match_object)
            if sol is None:
                raise NotImplementedError('The given ODE ' + str(eq) + ' cannot be solved by' + ' the hypergeometric method')
        return [sol]