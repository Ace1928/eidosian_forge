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
class LieGroup(SingleODESolver):
    """
    This hint implements the Lie group method of solving first order differential
    equations. The aim is to convert the given differential equation from the
    given coordinate system into another coordinate system where it becomes
    invariant under the one-parameter Lie group of translations. The converted
    ODE can be easily solved by quadrature. It makes use of the
    :py:meth:`sympy.solvers.ode.infinitesimals` function which returns the
    infinitesimals of the transformation.

    The coordinates `r` and `s` can be found by solving the following Partial
    Differential Equations.

    .. math :: \\xi\\frac{\\partial r}{\\partial x} + \\eta\\frac{\\partial r}{\\partial y}
                  = 0

    .. math :: \\xi\\frac{\\partial s}{\\partial x} + \\eta\\frac{\\partial s}{\\partial y}
                  = 1

    The differential equation becomes separable in the new coordinate system

    .. math :: \\frac{ds}{dr} = \\frac{\\frac{\\partial s}{\\partial x} +
                 h(x, y)\\frac{\\partial s}{\\partial y}}{
                 \\frac{\\partial r}{\\partial x} + h(x, y)\\frac{\\partial r}{\\partial y}}

    After finding the solution by integration, it is then converted back to the original
    coordinate system by substituting `r` and `s` in terms of `x` and `y` again.

    Examples
    ========

    >>> from sympy import Function, dsolve, exp, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x) + 2*x*f(x) - x*exp(-x**2), f(x),
    ... hint='lie_group'))
           /      2\\    2
           |     x |  -x
    f(x) = |C1 + --|*e
           \\     2 /


    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """
    hint = 'lie_group'
    has_integral = False

    def _has_additional_params(self):
        return 'xi' in self.ode_problem.params and 'eta' in self.ode_problem.params

    def _matches(self):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f(x).diff(x)
        y = Dummy('y')
        d = Wild('d', exclude=[df, f(x).diff(x, 2)])
        e = Wild('e', exclude=[df])
        does_match = False
        if self._has_additional_params() and order == 1:
            xi = self.ode_problem.params['xi']
            eta = self.ode_problem.params['eta']
            self.r3 = {'xi': xi, 'eta': eta}
            r = collect(eq, df, exact=True).match(d + e * df)
            if r:
                r['d'] = d
                r['e'] = e
                r['y'] = y
                r[d] = r[d].subs(f(x), y)
                r[e] = r[e].subs(f(x), y)
                self.r3.update(r)
            does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool=True):
        eq = self.ode_problem.eq
        x = self.ode_problem.sym
        func = self.ode_problem.func
        order = self.ode_problem.order
        df = func.diff(x)
        try:
            eqsol = solve(eq, df)
        except NotImplementedError:
            eqsol = []
        desols = []
        for s in eqsol:
            sol = _ode_lie_group(s, func, order, match=self.r3)
            if sol:
                desols.extend(sol)
        if desols == []:
            raise NotImplementedError('The given ODE ' + str(eq) + ' cannot be solved by' + ' the lie group method')
        return desols