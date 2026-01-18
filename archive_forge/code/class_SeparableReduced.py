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
class SeparableReduced(Separable):
    """
    Solves a differential equation that can be reduced to the separable form.

    The general form of this equation is

    .. math:: y' + (y/x) H(x^n y) = 0\\text{}.

    This can be solved by substituting `u(y) = x^n y`.  The equation then
    reduces to the separable form `\\frac{u'}{u (\\mathrm{power} - H(u))} -
    \\frac{1}{x} = 0`.

    The general solution is:

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x, n
        >>> f, g = map(Function, ['f', 'g'])
        >>> genform = f(x).diff(x) + (f(x)/x)*g(x**n*f(x))
        >>> pprint(genform)
                         / n     \\
        d          f(x)*g\\x *f(x)/
        --(f(x)) + ---------------
        dx                x
        >>> pprint(dsolve(genform, hint='separable_reduced'))
         n
        x *f(x)
          /
         |
         |         1
         |    ------------ dy = C1 + log(x)
         |    y*(n - g(y))
         |
         /

    See Also
    ========
    :obj:`sympy.solvers.ode.single.Separable`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = (x - x**2*f(x))*d - f(x)
    >>> dsolve(eq, hint='separable_reduced')
    [Eq(f(x), (1 - sqrt(C1*x**2 + 1))/x), Eq(f(x), (sqrt(C1*x**2 + 1) + 1)/x)]
    >>> pprint(dsolve(eq, hint='separable_reduced'))
                   ___________            ___________
                  /     2                /     2
            1 - \\/  C1*x  + 1          \\/  C1*x  + 1  + 1
    [f(x) = ------------------, f(x) = ------------------]
                    x                          x

    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = 'separable_reduced'
    has_integral = True
    order = [1]

    def _degree(self, expr, x):
        for val in expr:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    return val.as_base_exp()[1]
                elif val == x:
                    return val.as_base_exp()[1]
                else:
                    return self._degree(val.args, x)
        return 0

    def _powers(self, expr):
        pows = set()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.y = Dummy('y')
        if isinstance(expr, Add):
            exprs = expr.atoms(Add)
        elif isinstance(expr, Mul):
            exprs = expr.atoms(Mul)
        elif isinstance(expr, Pow):
            exprs = expr.atoms(Pow)
        else:
            exprs = {expr}
        for arg in exprs:
            if arg.has(x):
                _, u = arg.as_independent(x, fx)
                pow = self._degree((u.subs(fx, self.y),), x) / self._degree((u.subs(fx, self.y),), self.y)
                pows.add(pow)
        return pows

    def _verify(self, fx):
        num, den = self.wilds_match()
        x = self.ode_problem.sym
        factor = simplify(x / fx * num / den)
        num, dem = factor.as_numer_denom()
        num = expand(num)
        dem = expand(dem)
        pows = self._powers(num)
        pows.update(self._powers(dem))
        pows = list(pows)
        if len(pows) == 1 and pows[0] != zoo:
            self.t = Dummy('t')
            self.r2 = {'t': self.t}
            num = num.subs(x ** pows[0] * fx, self.t)
            dem = dem.subs(x ** pows[0] * fx, self.t)
            test = num / dem
            free = test.free_symbols
            if len(free) == 1 and free.pop() == self.t:
                self.r2.update({'power': pows[0], 'u': test})
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = self.r2['u'].subs(self.r2['t'], self.y)
        ycoeff = 1 / (self.y * (self.r2['power'] - u))
        m1 = {self.y: 1, x: -1 / x, 'coeff': 1}
        m2 = {self.y: ycoeff, x: 1, 'coeff': 1}
        return (m1, m2, x, x ** self.r2['power'] * fx)