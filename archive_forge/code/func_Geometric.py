from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import (polylog, zeta)
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import _value_check, is_random
def Geometric(name, p):
    """
    Create a discrete random variable with a Geometric distribution.

    Explanation
    ===========

    The density of the Geometric distribution is given by

    .. math::
        f(k) := p (1 - p)^{k - 1}

    Parameters
    ==========

    p : A probability between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Geometric, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Geometric("x", p)

    >>> density(X)(z)
    (5/4)**(1 - z)/5

    >>> E(X)
    5

    >>> variance(X)
    20

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Geometric_distribution
    .. [2] https://mathworld.wolfram.com/GeometricDistribution.html

    """
    return rv(name, GeometricDistribution, p)