from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda, PoleError
from sympy.core.numbers import (I, nan, oo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Union)
from sympy.solvers.solveset import solveset
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats.rv import (RandomDomain, SingleDomain, ConditionalDomain, is_random,
class SingleContinuousPSpace(ContinuousPSpace, SinglePSpace):
    """
    A continuous probability space over a single univariate variable.

    These consist of a Symbol and a SingleContinuousDistribution

    This class is normally accessed through the various random variable
    functions, Normal, Exponential, Uniform, etc....
    """

    @property
    def set(self):
        return self.distribution.set

    @property
    def domain(self):
        return SingleContinuousDomain(sympify(self.symbol), self.set)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method.

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        rvs = rvs or (self.value,)
        if self.value not in rvs:
            return expr
        expr = _sympify(expr)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})
        x = self.value.symbol
        try:
            return self.distribution.expectation(expr, x, evaluate=evaluate, **kwargs)
        except PoleError:
            return Integral(expr * self.pdf, (x, self.set), **kwargs)

    def compute_cdf(self, expr, **kwargs):
        if expr == self.value:
            z = Dummy('z', real=True)
            return Lambda(z, self.distribution.cdf(z, **kwargs))
        else:
            return ContinuousPSpace.compute_cdf(self, expr, **kwargs)

    def compute_characteristic_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy('t', real=True)
            return Lambda(t, self.distribution.characteristic_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_characteristic_function(self, expr, **kwargs)

    def compute_moment_generating_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy('t', real=True)
            return Lambda(t, self.distribution.moment_generating_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_moment_generating_function(self, expr, **kwargs)

    def compute_density(self, expr, **kwargs):
        if expr == self.value:
            return self.density
        y = Dummy('y', real=True)
        gs = solveset(expr - y, self.value, S.Reals)
        if isinstance(gs, Intersection) and S.Reals in gs.args:
            gs = list(gs.args[1])
        if not gs:
            raise ValueError('Can not solve %s for %s' % (expr, self.value))
        fx = self.compute_density(self.value)
        fy = sum((fx(g) * abs(g.diff(y)) for g in gs))
        return Lambda(y, fy)

    def compute_quantile(self, expr, **kwargs):
        if expr == self.value:
            p = Dummy('p', real=True)
            return Lambda(p, self.distribution.quantile(p, **kwargs))
        else:
            return ContinuousPSpace.compute_quantile(self, expr, **kwargs)