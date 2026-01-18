from __future__ import annotations
from functools import reduce
from sympy.core import S, sympify, Dummy, Mod
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
from sympy.core.numbers import Integer, pi, I
from sympy.core.relational import Eq
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.ntheory import sieve
from sympy.polys.polytools import Poly
from math import factorial as _factorial, prod, sqrt as _sqrt
class RisingFactorial(CombinatorialFunction):
    """
    Rising factorial (also called Pochhammer symbol [1]_) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by:

    .. math:: \\texttt{rf(y, k)} = (x)^k = x \\cdot (x+1) \\cdots (x+k-1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or visit https://mathworld.wolfram.com/RisingFactorial.html page.

    When `x` is a `~.Poly` instance of degree $\\ge 1$ with a single variable,
    `(x)^k = x(y) \\cdot x(y+1) \\cdots x(y+k-1)`, where `y` is the
    variable of `x`. This is as described in [2]_.

    Examples
    ========

    >>> from sympy import rf, Poly
    >>> from sympy.abc import x
    >>> rf(x, 0)
    1
    >>> rf(1, 5)
    120
    >>> rf(x, 5) == x*(1 + x)*(2 + x)*(3 + x)*(4 + x)
    True
    >>> rf(Poly(x**3, x), 2)
    Poly(x**6 + 3*x**5 + 3*x**4 + x**3, x, domain='ZZ')

    Rewriting is complicated unless the relationship between
    the arguments is known, but rising factorial can
    be rewritten in terms of gamma, factorial, binomial,
    and falling factorial.

    >>> from sympy import Symbol, factorial, ff, binomial, gamma
    >>> n = Symbol('n', integer=True, positive=True)
    >>> R = rf(n, n + 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  R.rewrite(i)
    ...
    RisingFactorial(n, n + 2)
    FallingFactorial(2*n + 1, n + 2)
    factorial(2*n + 1)/factorial(n - 1)
    binomial(2*n + 1, n + 2)*factorial(n + 2)
    gamma(2*n + 2)/gamma(n)

    See Also
    ========

    factorial, factorial2, FallingFactorial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pochhammer_symbol
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    """

    @classmethod
    def eval(cls, x, k):
        x = sympify(x)
        k = sympify(k)
        if x is S.NaN or k is S.NaN:
            return S.NaN
        elif x is S.One:
            return factorial(k)
        elif k.is_Integer:
            if k.is_zero:
                return S.One
            elif k.is_positive:
                if x is S.Infinity:
                    return S.Infinity
                elif x is S.NegativeInfinity:
                    if k.is_odd:
                        return S.NegativeInfinity
                    else:
                        return S.Infinity
                elif isinstance(x, Poly):
                    gens = x.gens
                    if len(gens) != 1:
                        raise ValueError('rf only defined for polynomials on one generator')
                    else:
                        return reduce(lambda r, i: r * x.shift(i), range(int(k)), 1)
                else:
                    return reduce(lambda r, i: r * (x + i), range(int(k)), 1)
            elif x is S.Infinity:
                return S.Infinity
            elif x is S.NegativeInfinity:
                return S.Infinity
            elif isinstance(x, Poly):
                gens = x.gens
                if len(gens) != 1:
                    raise ValueError('rf only defined for polynomials on one generator')
                else:
                    return 1 / reduce(lambda r, i: r * x.shift(-i), range(1, abs(int(k)) + 1), 1)
            else:
                return 1 / reduce(lambda r, i: r * (x - i), range(1, abs(int(k)) + 1), 1)
        if k.is_integer == False:
            if x.is_integer and x.is_negative:
                return S.Zero

    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        if not piecewise:
            if (x <= 0) == True:
                return S.NegativeOne ** k * gamma(1 - x) / gamma(-k - x + 1)
            return gamma(x + k) / gamma(x)
        return Piecewise((gamma(x + k) / gamma(x), x > 0), (S.NegativeOne ** k * gamma(1 - x) / gamma(-k - x + 1), True))

    def _eval_rewrite_as_FallingFactorial(self, x, k, **kwargs):
        return FallingFactorial(x + k - 1, k)

    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        if x.is_integer and k.is_integer:
            return Piecewise((factorial(k + x - 1) / factorial(x - 1), x > 0), (S.NegativeOne ** k * factorial(-x) / factorial(-k - x), True))

    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        if k.is_integer:
            return factorial(k) * binomial(x + k - 1, k)

    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        if limitvar:
            k_lim = k.subs(limitvar, S.Infinity)
            if k_lim is S.Infinity:
                return gamma(x + k).rewrite('tractable', deep=True) / gamma(x)
            elif k_lim is S.NegativeInfinity:
                return S.NegativeOne ** k * gamma(1 - x) / gamma(-k - x + 1).rewrite('tractable', deep=True)
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    def _eval_is_integer(self):
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer, self.args[1].is_nonnegative))