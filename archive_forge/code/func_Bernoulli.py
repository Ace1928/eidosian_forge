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
def Bernoulli(name, p, succ=1, fail=0):
    """
    Create a Finite Random Variable representing a Bernoulli process.

    Parameters
    ==========

    p : Rational number between 0 and 1
       Represents probability of success
    succ : Integer/symbol/string
       Represents event of success
    fail : Integer/symbol/string
       Represents event of failure

    Examples
    ========

    >>> from sympy.stats import Bernoulli, density
    >>> from sympy import S

    >>> X = Bernoulli('X', S(3)/4) # 1-0 Bernoulli variable, probability = 3/4
    >>> density(X).dict
    {0: 1/4, 1: 3/4}

    >>> X = Bernoulli('X', S.Half, 'Heads', 'Tails') # A fair coin toss
    >>> density(X).dict
    {Heads: 1/2, Tails: 1/2}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_distribution
    .. [2] https://mathworld.wolfram.com/BernoulliDistribution.html

    """
    return rv(name, BernoulliDistribution, p, succ, fail)