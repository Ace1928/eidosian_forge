from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import (Integer, Rational)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Or
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.stats.frv import (SingleFiniteDistribution,
from sympy.stats.rv import _value_check, Density, is_random
from sympy.utilities.iterables import multiset
from sympy.utilities.misc import filldedent
class HypergeometricDistribution(SingleFiniteDistribution):
    _argnames = ('N', 'm', 'n')

    @staticmethod
    def check(n, N, m):
        _value_check((N.is_integer, N.is_nonnegative), "'N' must be nonnegative integer. N = %s." % str(n))
        _value_check((n.is_integer, n.is_nonnegative), "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check((m.is_integer, m.is_nonnegative), "'m' must be nonnegative integer. m = %s." % str(n))

    @property
    def is_symbolic(self):
        return not all((x.is_number for x in (self.N, self.m, self.n)))

    @property
    def high(self):
        return Piecewise((self.n, Lt(self.n, self.m) != False), (self.m, True))

    @property
    def low(self):
        return Piecewise((0, Gt(0, self.n + self.m - self.N) != False), (self.n + self.m - self.N, True))

    @property
    def set(self):
        N, m, n = (self.N, self.m, self.n)
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(self.low, self.high))
        return set(range(max(0, n + m - N), min(n, m) + 1))

    def pmf(self, k):
        N, m, n = (self.N, self.m, self.n)
        return S(binomial(m, k) * binomial(N - m, n - k)) / binomial(N, n)