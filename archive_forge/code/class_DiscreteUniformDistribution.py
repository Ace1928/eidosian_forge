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
class DiscreteUniformDistribution(SingleFiniteDistribution):

    @staticmethod
    def check(*args):
        if len(set(args)) != len(args):
            weights = multiset(args)
            n = Integer(len(args))
            for k in weights:
                weights[k] /= n
            raise ValueError(filldedent('\n                Repeated args detected but set expected. For a\n                distribution having different weights for each\n                item use the following:') + '\nS("FiniteRV(%s, %s)")' % ("'X'", weights))

    @property
    def p(self):
        return Rational(1, len(self.args))

    @property
    @cacheit
    def dict(self):
        return {k: self.p for k in self.set}

    @property
    def set(self):
        return set(self.args)

    def pmf(self, x):
        if x in self.args:
            return self.p
        else:
            return S.Zero