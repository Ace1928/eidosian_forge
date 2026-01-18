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
def Weibull(name, alpha, beta):
    """
    Create a continuous random variable with a Weibull distribution.

    Explanation
    ===========

    The density of the Weibull distribution is given by

    .. math::
        f(x) := \\begin{cases}
                  \\frac{k}{\\lambda}\\left(\\frac{x}{\\lambda}\\right)^{k-1}
                  e^{-(x/\\lambda)^{k}} & x\\geq0\\\\
                  0 & x<0
                \\end{cases}

    Parameters
    ==========

    lambda : Real number, $\\lambda > 0$, a scale
    k : Real number, $k > 0$, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Weibull, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> l = Symbol("lambda", positive=True)
    >>> k = Symbol("k", positive=True)
    >>> z = Symbol("z")

    >>> X = Weibull("x", l, k)

    >>> density(X)(z)
    k*(z/lambda)**(k - 1)*exp(-(z/lambda)**k)/lambda

    >>> simplify(E(X))
    lambda*gamma(1 + 1/k)

    >>> simplify(variance(X))
    lambda**2*(-gamma(1 + 1/k)**2 + gamma(1 + 2/k))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Weibull_distribution
    .. [2] https://mathworld.wolfram.com/WeibullDistribution.html

    """
    return rv(name, WeibullDistribution, (alpha, beta))