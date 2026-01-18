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
class BetaBinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'alpha', 'beta')

    @staticmethod
    def check(n, alpha, beta):
        _value_check((n.is_integer, n.is_nonnegative), "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check(alpha > 0, "'alpha' must be: alpha > 0 . alpha = %s" % str(alpha))
        _value_check(beta > 0, "'beta' must be: beta > 0 . beta = %s" % str(beta))

    @property
    def high(self):
        return self.n

    @property
    def low(self):
        return S.Zero

    @property
    def is_symbolic(self):
        return not self.n.is_number

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(map(Integer, range(self.n + 1)))

    def pmf(self, k):
        n, a, b = (self.n, self.alpha, self.beta)
        return binomial(n, k) * beta_fn(k + a, n - k + b) / beta_fn(a, b)