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
def Lomax(name, alpha, lamda):
    """
    Create a continuous random variable with a Lomax distribution.

    Explanation
    ===========

    The density of the Lomax distribution is given by

    .. math::
        f(x) := \\frac{\\alpha}{\\lambda}\\left[1+\\frac{x}{\\lambda}\\right]^{-(\\alpha+1)}

    Parameters
    ==========

    alpha : Real Number, `\\alpha > 0`
        Shape parameter
    lamda : Real Number, `\\lambda > 0`
        Scale parameter

    Examples
    ========

    >>> from sympy.stats import Lomax, density, cdf, E
    >>> from sympy import symbols
    >>> a, l = symbols('a, l', positive=True)
    >>> X = Lomax('X', a, l)
    >>> x = symbols('x')
    >>> density(X)(x)
    a*(1 + x/l)**(-a - 1)/l
    >>> cdf(X)(x)
    Piecewise((1 - 1/(1 + x/l)**a, x >= 0), (0, True))
    >>> a = 2
    >>> X = Lomax('X', a, l)
    >>> E(X)
    l

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lomax_distribution

    """
    return rv(name, LomaxDistribution, (alpha, lamda))