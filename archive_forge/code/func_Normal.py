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
def Normal(name, mean, std):
    """
    Create a continuous random variable with a Normal distribution.

    Explanation
    ===========

    The density of the Normal distribution is given by

    .. math::
        f(x) := \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{ -\\frac{(x-\\mu)^2}{2\\sigma^2} }

    Parameters
    ==========

    mu : Real number or a list representing the mean or the mean vector
    sigma : Real number or a positive definite square matrix,
         :math:`\\sigma^2 > 0`, the variance

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Normal, density, E, std, cdf, skewness, quantile, marginal_distribution
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu")
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")
    >>> y = Symbol("y")
    >>> p = Symbol("p")
    >>> X = Normal("x", mu, sigma)

    >>> density(X)(z)
    sqrt(2)*exp(-(-mu + z)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)

    >>> C = simplify(cdf(X))(z) # it needs a little more help...
    >>> pprint(C, use_unicode=False)
       /  ___          \\
       |\\/ 2 *(-mu + z)|
    erf|---------------|
       \\    2*sigma    /   1
    -------------------- + -
             2             2

    >>> quantile(X)(p)
    mu + sqrt(2)*sigma*erfinv(2*p - 1)

    >>> simplify(skewness(X))
    0

    >>> X = Normal("x", 0, 1) # Mean 0, standard deviation 1
    >>> density(X)(z)
    sqrt(2)*exp(-z**2/2)/(2*sqrt(pi))

    >>> E(2*X + 1)
    1

    >>> simplify(std(2*X + 1))
    2

    >>> m = Normal('X', [1, 2], [[2, 1], [1, 2]])
    >>> pprint(density(m)(y, z), use_unicode=False)
              2          2
             y    y*z   z
           - -- + --- - -- + z - 1
      ___    3     3    3
    \\/ 3 *e
    ------------------------------
                 6*pi

    >>> marginal_distribution(m, m[0])(1)
     1/(2*sqrt(pi))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] https://mathworld.wolfram.com/NormalDistributionFunction.html

    """
    if isinstance(mean, list) or (getattr(mean, 'is_Matrix', False) and isinstance(std, list)) or getattr(std, 'is_Matrix', False):
        from sympy.stats.joint_rv_types import MultivariateNormal
        return MultivariateNormal(name, mean, std)
    return rv(name, NormalDistribution, (mean, std))