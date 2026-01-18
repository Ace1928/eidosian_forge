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
def Uniform(name, left, right):
    """
    Create a continuous random variable with a uniform distribution.

    Explanation
    ===========

    The density of the uniform distribution is given by

    .. math::
        f(x) := \\begin{cases}
                  \\frac{1}{b - a} & \\text{for } x \\in [a,b]  \\\\
                  0               & \\text{otherwise}
                \\end{cases}

    with :math:`x \\in [a,b]`.

    Parameters
    ==========

    a : Real number, :math:`-\\infty < a`, the left boundary
    b : Real number, :math:`a < b < \\infty`, the right boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Uniform, density, cdf, E, variance
    >>> from sympy import Symbol, simplify

    >>> a = Symbol("a", negative=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Uniform("x", a, b)

    >>> density(X)(z)
    Piecewise((1/(-a + b), (b >= z) & (a <= z)), (0, True))

    >>> cdf(X)(z)
    Piecewise((0, a > z), ((-a + z)/(-a + b), b >= z), (1, True))

    >>> E(X)
    a/2 + b/2

    >>> simplify(variance(X))
    a**2/12 - a*b/6 + b**2/12

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Uniform_distribution_%28continuous%29
    .. [2] https://mathworld.wolfram.com/UniformDistribution.html

    """
    return rv(name, UniformDistribution, (left, right))