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
class SingleODESolver:
    """
    Base class for Single ODE solvers.

    Subclasses should implement the _matches and _get_general_solution
    methods. This class is not intended to be instantiated directly but its
    subclasses are as part of dsolve.

    Examples
    ========

    You can use a subclass of SingleODEProblem to solve a particular type of
    ODE. We first define a particular ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now we solve this problem using the NthAlgebraic solver which is a
    subclass of SingleODESolver:

    >>> from sympy.solvers.ode.single import NthAlgebraic, SingleODEProblem
    >>> problem = SingleODEProblem(eq, f(x), x)
    >>> solver = NthAlgebraic(problem)
    >>> solver.get_general_solution()
    [Eq(f(x), _C*x + _C)]

    The normal way to solve an ODE is to use dsolve (which would use
    NthAlgebraic and other solvers internally). When using dsolve a number of
    other things are done such as evaluating integrals, simplifying the
    solution and renumbering the constants:

    >>> from sympy import dsolve
    >>> dsolve(eq, hint='nth_algebraic')
    Eq(f(x), C1 + C2*x)
    """
    hint: ClassVar[str]
    has_integral: ClassVar[bool]
    ode_problem = None
    _matched: bool | None = None
    order: list | None = None

    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def matches(self) -> bool:
        if self.order is not None and self.ode_problem.order not in self.order:
            self._matched = False
            return self._matched
        if self._matched is None:
            self._matched = self._matches()
        return self._matched

    def get_general_solution(self, *, simplify: bool=True) -> list[Equality]:
        if not self.matches():
            msg = '%s solver cannot solve:\n%s'
            raise ODEMatchError(msg % (self.hint, self.ode_problem.eq))
        return self._get_general_solution(simplify_flag=simplify)

    def _matches(self) -> bool:
        msg = 'Subclasses of SingleODESolver should implement matches.'
        raise NotImplementedError(msg)

    def _get_general_solution(self, *, simplify_flag: bool=True) -> list[Equality]:
        msg = 'Subclasses of SingleODESolver should implement get_general_solution.'
        raise NotImplementedError(msg)