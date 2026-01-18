from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, Function
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly
class zeta(Function):
    """
    Hurwitz zeta function (or Riemann zeta function).

    Explanation
    ===========

    For $\\operatorname{Re}(a) > 0$ and $\\operatorname{Re}(s) > 1$, this
    function is defined as

    .. math:: \\zeta(s, a) = \\sum_{n=0}^\\infty \\frac{1}{(n + a)^s},

    where the standard choice of argument for $n + a$ is used. For fixed
    $a$ not a nonpositive integer the Hurwitz zeta function admits a
    meromorphic continuation to all of $\\mathbb{C}$; it is an unbranched
    function with a simple pole at $s = 1$.

    The Hurwitz zeta function is a special case of the Lerch transcendent:

    .. math:: \\zeta(s, a) = \\Phi(1, s, a).

    This formula defines an analytic continuation for all possible values of
    $s$ and $a$ (also $\\operatorname{Re}(a) < 0$), see the documentation of
    :class:`lerchphi` for a description of the branching behavior.

    If no value is passed for $a$ a default value of $a = 1$ is assumed,
    yielding the Riemann zeta function.

    Examples
    ========

    For $a = 1$ the Hurwitz zeta function reduces to the famous Riemann
    zeta function:

    .. math:: \\zeta(s, 1) = \\zeta(s) = \\sum_{n=1}^\\infty \\frac{1}{n^s}.

    >>> from sympy import zeta
    >>> from sympy.abc import s
    >>> zeta(s, 1)
    zeta(s)
    >>> zeta(s)
    zeta(s)

    The Riemann zeta function can also be expressed using the Dirichlet eta
    function:

    >>> from sympy import dirichlet_eta
    >>> zeta(s).rewrite(dirichlet_eta)
    dirichlet_eta(s)/(1 - 2**(1 - s))

    The Riemann zeta function at nonnegative even and negative integer
    values is related to the Bernoulli numbers and polynomials:

    >>> zeta(2)
    pi**2/6
    >>> zeta(4)
    pi**4/90
    >>> zeta(0)
    -1/2
    >>> zeta(-1)
    -1/12
    >>> zeta(-4)
    0

    The specific formulae are:

    .. math:: \\zeta(2n) = -\\frac{(2\\pi i)^{2n} B_{2n}}{2(2n)!}
    .. math:: \\zeta(-n,a) = -\\frac{B_{n+1}(a)}{n+1}

    No closed-form expressions are known at positive odd integers, but
    numerical evaluation is possible:

    >>> zeta(3).n()
    1.20205690315959

    The derivative of $\\zeta(s, a)$ with respect to $a$ can be computed:

    >>> from sympy.abc import a
    >>> zeta(s, a).diff(a)
    -s*zeta(s + 1, a)

    However the derivative with respect to $s$ has no useful closed form
    expression:

    >>> zeta(s, a).diff(s)
    Derivative(zeta(s, a), s)

    The Hurwitz zeta function can be expressed in terms of the Lerch
    transcendent, :class:`~.lerchphi`:

    >>> from sympy import lerchphi
    >>> zeta(s, a).rewrite(lerchphi)
    lerchphi(1, s, a)

    See Also
    ========

    dirichlet_eta, lerchphi, polylog

    References
    ==========

    .. [1] https://dlmf.nist.gov/25.11
    .. [2] https://en.wikipedia.org/wiki/Hurwitz_zeta_function

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        elif s is S.NaN or a is S.NaN:
            return S.NaN
        elif s is S.One:
            return S.ComplexInfinity
        elif s is S.Infinity:
            return S.One
        elif a is S.Infinity:
            return S.Zero
        sint = s.is_Integer
        if a is None:
            a = S.One
        if sint and s.is_nonpositive:
            return bernoulli(1 - s, a) / (s - 1)
        elif a is S.One:
            if sint and s.is_even:
                return -(2 * pi * I) ** s * bernoulli(s) / (2 * factorial(s))
        elif sint and a.is_Integer and a.is_positive:
            return cls(s) - harmonic(a - 1, s)
        elif a.is_Integer and a.is_nonpositive and (s.is_integer is False or s.is_nonpositive is False):
            return S.NaN

    def _eval_rewrite_as_bernoulli(self, s, a=1, **kwargs):
        if a == 1 and s.is_integer and s.is_nonnegative and s.is_even:
            return -(2 * pi * I) ** s * bernoulli(s) / (2 * factorial(s))
        return bernoulli(1 - s, a) / (s - 1)

    def _eval_rewrite_as_dirichlet_eta(self, s, a=1, **kwargs):
        if a != 1:
            return self
        s = self.args[0]
        return dirichlet_eta(s) / (1 - 2 ** (1 - s))

    def _eval_rewrite_as_lerchphi(self, s, a=1, **kwargs):
        return lerchphi(1, s, a)

    def _eval_is_finite(self):
        arg_is_one = (self.args[0] - 1).is_zero
        if arg_is_one is not None:
            return not arg_is_one

    def _eval_expand_func(self, **hints):
        s = self.args[0]
        a = self.args[1] if len(self.args) > 1 else S.One
        if a.is_integer:
            if a.is_positive:
                return zeta(s) - harmonic(a - 1, s)
            if a.is_nonpositive and (s.is_integer is False or s.is_nonpositive is False):
                return S.NaN
        return self

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (1,)
        if argindex == 2:
            return -s * zeta(s + 1, a)
        else:
            raise ArgumentIndexError

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (S.One,)
        try:
            c, e = a.leadterm(x)
        except NotImplementedError:
            return self
        if e.is_negative and (not s.is_positive):
            raise NotImplementedError
        return super(zeta, self)._eval_as_leading_term(x, logx, cdir)