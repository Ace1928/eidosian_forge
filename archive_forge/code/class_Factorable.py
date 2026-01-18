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
class Factorable(SingleODESolver):
    """
        Solves equations having a solvable factor.

        This function is used to solve the equation having factors. Factors may be of type algebraic or ode. It
        will try to solve each factor independently. Factors will be solved by calling dsolve. We will return the
        list of solutions.

        Examples
        ========

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x
        >>> f = Function('f')
        >>> eq = (f(x)**2-4)*(f(x).diff(x)+f(x))
        >>> pprint(dsolve(eq, f(x)))
                                        -x
        [f(x) = 2, f(x) = -2, f(x) = C1*e  ]


        """
    hint = 'factorable'
    has_integral = False

    def _matches(self):
        eq_orig = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        df = f(x).diff(x)
        self.eqs = []
        eq = eq_orig.collect(f(x), func=cancel)
        eq = fraction(factor(eq))[0]
        factors = Mul.make_args(factor(eq))
        roots = [fac.as_base_exp() for fac in factors if len(fac.args) != 0]
        if len(roots) > 1 or roots[0][1] > 1:
            for base, expo in roots:
                if base.has(f(x)):
                    self.eqs.append(base)
            if len(self.eqs) > 0:
                return True
        roots = solve(eq, df)
        if len(roots) > 0:
            self.eqs = [df - root for root in roots]
            matches = self.eqs != [eq_orig]
            return matches
        for i in factors:
            if i.has(f(x)):
                self.eqs.append(i)
        return len(self.eqs) > 0 and len(factors) > 1

    def _get_general_solution(self, *, simplify_flag: bool=True):
        func = self.ode_problem.func.func
        x = self.ode_problem.sym
        eqns = self.eqs
        sols = []
        for eq in eqns:
            try:
                sol = dsolve(eq, func(x))
            except NotImplementedError:
                continue
            else:
                if isinstance(sol, list):
                    sols.extend(sol)
                else:
                    sols.append(sol)
        if sols == []:
            raise NotImplementedError('The given ODE ' + str(eq) + ' cannot be solved by' + ' the factorable group method')
        return sols