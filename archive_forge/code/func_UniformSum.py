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
def UniformSum(name, n):
    """
    Create a continuous random variable with an Irwin-Hall distribution.

    Explanation
    ===========

    The probability distribution function depends on a single parameter
    $n$ which is an integer.

    The density of the Irwin-Hall distribution is given by

    .. math ::
        f(x) := \\frac{1}{(n-1)!}\\sum_{k=0}^{\\left\\lfloor x\\right\\rfloor}(-1)^k
                \\binom{n}{k}(x-k)^{n-1}

    Parameters
    ==========

    n : A positive integer, `n > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import UniformSum, density, cdf
    >>> from sympy import Symbol, pprint

    >>> n = Symbol("n", integer=True)
    >>> z = Symbol("z")

    >>> X = UniformSum("x", n)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    floor(z)
      ___
      \\  `
       \\         k         n - 1 /n\\
        )    (-1) *(-k + z)     *| |
       /                         \\k/
      /__,
     k = 0
    --------------------------------
                (n - 1)!

    >>> cdf(X)(z)
    Piecewise((0, z < 0), (Sum((-1)**_k*(-_k + z)**n*binomial(n, _k),
                    (_k, 0, floor(z)))/factorial(n), n >= z), (1, True))


    Compute cdf with specific 'x' and 'n' values as follows :
    >>> cdf(UniformSum("x", 5), evaluate=False)(2).doit()
    9/40

    The argument evaluate=False prevents an attempt at evaluation
    of the sum for general n, before the argument 2 is passed.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Uniform_sum_distribution
    .. [2] https://mathworld.wolfram.com/UniformSumDistribution.html

    """
    return rv(name, UniformSumDistribution, (n,))