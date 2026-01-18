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
def Trapezoidal(name, a, b, c, d):
    """
    Create a continuous random variable with a trapezoidal distribution.

    Explanation
    ===========

    The density of the trapezoidal distribution is given by

    .. math::
        f(x) := \\begin{cases}
                  0 & \\mathrm{for\\ } x < a, \\\\
                  \\frac{2(x-a)}{(b-a)(d+c-a-b)} & \\mathrm{for\\ } a \\le x < b, \\\\
                  \\frac{2}{d+c-a-b} & \\mathrm{for\\ } b \\le x < c, \\\\
                  \\frac{2(d-x)}{(d-c)(d+c-a-b)} & \\mathrm{for\\ } c \\le x < d, \\\\
                  0 & \\mathrm{for\\ } d < x.
                \\end{cases}

    Parameters
    ==========

    a : Real number, :math:`a < d`
    b : Real number, :math:`a \\le b < c`
    c : Real number, :math:`b < c \\le d`
    d : Real number

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Trapezoidal, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a")
    >>> b = Symbol("b")
    >>> c = Symbol("c")
    >>> d = Symbol("d")
    >>> z = Symbol("z")

    >>> X = Trapezoidal("x", a,b,c,d)

    >>> pprint(density(X)(z), use_unicode=False)
    /        -2*a + 2*z
    |-------------------------  for And(a <= z, b > z)
    |(-a + b)*(-a - b + c + d)
    |
    |           2
    |     --------------        for And(b <= z, c > z)
    <     -a - b + c + d
    |
    |        2*d - 2*z
    |-------------------------  for And(d >= z, c <= z)
    |(-c + d)*(-a - b + c + d)
    |
    \\            0                     otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trapezoidal_distribution

    """
    return rv(name, TrapezoidalDistribution, (a, b, c, d))