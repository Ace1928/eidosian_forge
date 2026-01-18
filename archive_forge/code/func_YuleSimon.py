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
def YuleSimon(name, rho):
    """
    Create a discrete random variable with a Yule-Simon distribution.

    Explanation
    ===========

    The density of the Yule-Simon distribution is given by

    .. math::
        f(k) := \\rho B(k, \\rho + 1)

    Parameters
    ==========

    rho : A positive value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import YuleSimon, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> p = 5
    >>> z = Symbol("z")

    >>> X = YuleSimon("x", p)

    >>> density(X)(z)
    5*beta(z, 6)

    >>> simplify(E(X))
    5/4

    >>> simplify(variance(X))
    25/48

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution

    """
    return rv(name, YuleSimonDistribution, rho)