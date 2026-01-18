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
def BetaBinomial(name, n, alpha, beta):
    """
    Create a Finite Random Variable representing a Beta-binomial distribution.

    Parameters
    ==========

    n : Positive Integer
      Represents number of trials
    alpha : Real positive number
    beta : Real positive number

    Examples
    ========

    >>> from sympy.stats import BetaBinomial, density

    >>> X = BetaBinomial('X', 2, 1, 1)
    >>> density(X).dict
    {0: 1/3, 1: 2*beta(2, 2), 2: 1/3}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution
    .. [2] https://mathworld.wolfram.com/BetaBinomialDistribution.html

    """
    return rv(name, BetaBinomialDistribution, n, alpha, beta)