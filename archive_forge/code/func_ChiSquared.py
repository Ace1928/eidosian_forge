from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random
def ChiSquared(name, k):
    """
    Create a continuous random variable with a Chi-squared distribution.

    Explanation
    ===========

    The density of the Chi-squared distribution is given by

    .. math::
        f(x) := \\frac{1}{2^{\\frac{k}{2}}\\Gamma\\left(\\frac{k}{2}\\right)}
                x^{\\frac{k}{2}-1} e^{-\\frac{x}{2}}

    with :math:`x \\geq 0`.

    Parameters
    ==========

    k : Positive integer
        The number of degrees of freedom.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiSquared, density, E, variance, moment
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True, positive=True)
    >>> z = Symbol("z")

    >>> X = ChiSquared("x", k)

    >>> density(X)(z)
    z**(k/2 - 1)*exp(-z/2)/(2**(k/2)*gamma(k/2))

    >>> E(X)
    k

    >>> variance(X)
    2*k

    >>> moment(X, 3)
    k**3 + 6*k**2 + 8*k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chi_squared_distribution
    .. [2] https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    """
    return rv(name, ChiSquaredDistribution, (k,))