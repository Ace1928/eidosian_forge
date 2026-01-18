from ..libmp.backend import xrange
from .calculus import defun
class levin_class:
    """
    This interface implements Levin's (nonlinear) sequence transformation for
    convergence acceleration and summation of divergent series. It performs
    better than the Shanks/Wynn-epsilon algorithm for logarithmic convergent
    or alternating divergent series.

    Let *A* be the series we want to sum:

    .. math ::

        A = \\sum_{k=0}^{\\infty} a_k

    Attention: all `a_k` must be non-zero!

    Let `s_n` be the partial sums of this series:

    .. math ::

        s_n = \\sum_{k=0}^n a_k.

    **Methods**

    Calling ``levin`` returns an object with the following methods.

    ``update(...)`` works with the list of individual terms `a_k` of *A*, and
    ``update_step(...)`` works with the list of partial sums `s_k` of *A*:

    .. code ::

        v, e = ...update([a_0, a_1,..., a_k])
        v, e = ...update_psum([s_0, s_1,..., s_k])

    ``step(...)`` works with the individual terms `a_k` and ``step_psum(...)``
    works with the partial sums `s_k`:

    .. code ::

        v, e = ...step(a_k)
        v, e = ...step_psum(s_k)

    *v* is the current estimate for *A*, and *e* is an error estimate which is
    simply the difference between the current estimate and the last estimate.
    One should not mix ``update``, ``update_psum``, ``step`` and ``step_psum``.

    **A word of caution**

    One can only hope for good results (i.e. convergence acceleration or
    resummation) if the `s_n` have some well defind asymptotic behavior for
    large `n` and are not erratic or random. Furthermore one usually needs very
    high working precision because of the numerical cancellation. If the working
    precision is insufficient, levin may produce silently numerical garbage.
    Furthermore even if the Levin-transformation converges, in the general case
    there is no proof that the result is mathematically sound. Only for very
    special classes of problems one can prove that the Levin-transformation
    converges to the expected result (for example Stieltjes-type integrals).
    Furthermore the Levin-transform is quite expensive (i.e. slow) in comparison
    to Shanks/Wynn-epsilon, Richardson & co.
    In summary one can say that the Levin-transformation is powerful but
    unreliable and that it may need a copious amount of working precision.

    The Levin transform has several variants differing in the choice of weights.
    Some variants are better suited for the possible flavours of convergence
    behaviour of *A* than other variants:

    .. code ::

       convergence behaviour   levin-u   levin-t   levin-v   shanks/wynn-epsilon

       logarithmic               +         -         +           -
       linear                    +         +         +           +
       alternating divergent     +         +         +           +

         "+" means the variant is suitable,"-" means the variant is not suitable;
         for comparison the Shanks/Wynn-epsilon transform is listed, too.

    The variant is controlled though the variant keyword (i.e. ``variant="u"``,
    ``variant="t"`` or ``variant="v"``). Overall "u" is probably the best choice.

    Finally it is possible to use the Sidi-S transform instead of the Levin transform
    by using the keyword ``method='sidi'``. The Sidi-S transform works better than the
    Levin transformation for some divergent series (see the examples).

    Parameters:

    .. code ::

       method      "levin" or "sidi" chooses either the Levin or the Sidi-S transformation
       variant     "u","t" or "v" chooses the weight variant.

    The Levin transform is also accessible through the nsum interface.
    ``method="l"`` or ``method="levin"`` select the normal Levin transform while
    ``method="sidi"``
    selects the Sidi-S transform. The variant is in both cases selected through the
    levin_variant keyword. The stepsize in :func:`~mpmath.nsum` must not be chosen too large, otherwise
    it will miss the point where the Levin transform converges resulting in numerical
    overflow/garbage. For highly divergent series a copious amount of working precision
    must be chosen.

    **Examples**

    First we sum the zeta function::

        >>> from mpmath import mp
        >>> mp.prec = 53
        >>> eps = mp.mpf(mp.eps)
        >>> with mp.extraprec(2 * mp.prec): # levin needs a high working precision
        ...     L = mp.levin(method = "levin", variant = "u")
        ...     S, s, n = [], 0, 1
        ...     while 1:
        ...         s += mp.one / (n * n)
        ...         n += 1
        ...         S.append(s)
        ...         v, e = L.update_psum(S)
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - mp.pi ** 2 / 6))
        0.0
        >>> w = mp.nsum(lambda n: 1 / (n*n), [1, mp.inf], method = "levin", levin_variant = "u")
        >>> print(mp.chop(v - w))
        0.0

    Now we sum the zeta function outside its range of convergence
    (attention: This does not work at the negative integers!)::

        >>> eps = mp.mpf(mp.eps)
        >>> with mp.extraprec(2 * mp.prec): # levin needs a high working precision
        ...     L = mp.levin(method = "levin", variant = "v")
        ...     A, n = [], 1
        ...     while 1:
        ...         s = mp.mpf(n) ** (2 + 3j)
        ...         n += 1
        ...         A.append(s)
        ...         v, e = L.update(A)
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - mp.zeta(-2-3j)))
        0.0
        >>> w = mp.nsum(lambda n: n ** (2 + 3j), [1, mp.inf], method = "levin", levin_variant = "v")
        >>> print(mp.chop(v - w))
        0.0

    Now we sum the divergent asymptotic expansion of an integral related to the
    exponential integral (see also [2] p.373). The Sidi-S transform works best here::

        >>> z = mp.mpf(10)
        >>> exact = mp.quad(lambda x: mp.exp(-x)/(1+x/z),[0,mp.inf])
        >>> # exact = z * mp.exp(z) * mp.expint(1,z) # this is the symbolic expression for the integral
        >>> eps = mp.mpf(mp.eps)
        >>> with mp.extraprec(2 * mp.prec): # high working precisions are mandatory for divergent resummation
        ...     L = mp.levin(method = "sidi", variant = "t")
        ...     n = 0
        ...     while 1:
        ...         s = (-1)**n * mp.fac(n) * z ** (-n)
        ...         v, e = L.step(s)
        ...         n += 1
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - exact))
        0.0
        >>> w = mp.nsum(lambda n: (-1) ** n * mp.fac(n) * z ** (-n), [0, mp.inf], method = "sidi", levin_variant = "t")
        >>> print(mp.chop(v - w))
        0.0

    Another highly divergent integral is also summable::

        >>> z = mp.mpf(2)
        >>> eps = mp.mpf(mp.eps)
        >>> exact = mp.quad(lambda x: mp.exp( -x * x / 2 - z * x ** 4), [0,mp.inf]) * 2 / mp.sqrt(2 * mp.pi)
        >>> # exact = mp.exp(mp.one / (32 * z)) * mp.besselk(mp.one / 4, mp.one / (32 * z)) / (4 * mp.sqrt(z * mp.pi)) # this is the symbolic expression for the integral
        >>> with mp.extraprec(7 * mp.prec):  # we need copious amount of precision to sum this highly divergent series
        ...     L = mp.levin(method = "levin", variant = "t")
        ...     n, s = 0, 0
        ...     while 1:
        ...         s += (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n))
        ...         n += 1
        ...         v, e = L.step_psum(s)
        ...         if e < eps:
        ...             break
        ...         if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - exact))
        0.0
        >>> w = mp.nsum(lambda n: (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n)),
        ...   [0, mp.inf], method = "levin", levin_variant = "t", workprec = 8*mp.prec, steps = [2] + [1 for x in xrange(1000)])
        >>> print(mp.chop(v - w))
        0.0

    These examples run with 15-20 decimal digits precision. For higher precision the
    working precision must be raised.

    **Examples for nsum**

    Here we calculate Euler's constant as the constant term in the Laurent
    expansion of `\\zeta(s)` at `s=1`. This sum converges extremly slowly because of
    the logarithmic convergence behaviour of the Dirichlet series for zeta::

        >>> mp.dps = 30
        >>> z = mp.mpf(10) ** (-10)
        >>> a = mp.nsum(lambda n: n**(-(1+z)), [1, mp.inf], method = "l") - 1 / z
        >>> print(mp.chop(a - mp.euler, tol = 1e-10))
        0.0

    The Sidi-S transform performs excellently for the alternating series of `\\log(2)`::

        >>> a = mp.nsum(lambda n: (-1)**(n-1) / n, [1, mp.inf], method = "sidi")
        >>> print(mp.chop(a - mp.log(2)))
        0.0

    Hypergeometric series can also be summed outside their range of convergence.
    The stepsize in :func:`~mpmath.nsum` must not be chosen too large, otherwise it will miss the
    point where the Levin transform converges resulting in numerical overflow/garbage::

        >>> z = 2 + 1j
        >>> exact = mp.hyp2f1(2 / mp.mpf(3), 4 / mp.mpf(3), 1 / mp.mpf(3), z)
        >>> f = lambda n: mp.rf(2 / mp.mpf(3), n) * mp.rf(4 / mp.mpf(3), n) * z**n / (mp.rf(1 / mp.mpf(3), n) * mp.fac(n))
        >>> v = mp.nsum(f, [0, mp.inf], method = "levin", steps = [10 for x in xrange(1000)])
        >>> print(mp.chop(exact-v))
        0.0

    References:

      [1] E.J. Weniger - "Nonlinear Sequence Transformations for the Acceleration of
          Convergence and the Summation of Divergent Series" arXiv:math/0306302

      [2] A. Sidi - "Pratical Extrapolation Methods"

      [3] H.H.H. Homeier - "Scalar Levin-Type Sequence Transformations" arXiv:math/0005209

    """

    def __init__(self, method='levin', variant='u'):
        self.variant = variant
        self.n = 0
        self.a0 = 0
        self.theta = 1
        self.A = []
        self.B = []
        self.last = 0
        self.last_s = False
        if method == 'levin':
            self.factor = self.factor_levin
        elif method == 'sidi':
            self.factor = self.factor_sidi
        else:
            raise ValueError('levin: unknown method "%s"' % method)

    def factor_levin(self, i):
        return (self.theta + i) * (self.theta + self.n - 1) ** (self.n - i - 2) / self.ctx.mpf(self.theta + self.n) ** (self.n - i - 1)

    def factor_sidi(self, i):
        return (self.theta + self.n - 1) * (self.theta + self.n - 2) / self.ctx.mpf((self.theta + 2 * self.n - i - 2) * (self.theta + 2 * self.n - i - 3))

    def run(self, s, a0, a1=0):
        if self.variant == 't':
            w = a0
        elif self.variant == 'u':
            w = a0 * (self.theta + self.n)
        elif self.variant == 'v':
            w = a0 * a1 / (a0 - a1)
        else:
            assert False, 'unknown variant'
        if w == 0:
            raise ValueError('levin: zero weight')
        self.A.append(s / w)
        self.B.append(1 / w)
        for i in range(self.n - 1, -1, -1):
            if i == self.n - 1:
                f = 1
            else:
                f = self.factor(i)
            self.A[i] = self.A[i + 1] - f * self.A[i]
            self.B[i] = self.B[i + 1] - f * self.B[i]
        self.n += 1

    def update_psum(self, S):
        """
        This routine applies the convergence acceleration to the list of partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k, k = 0..n)

        v, e = ...update_psum([s_0, s_1,..., s_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
        if self.variant != 'v':
            if self.n == 0:
                self.run(S[0], S[0])
            while self.n < len(S):
                self.run(S[self.n], S[self.n] - S[self.n - 1])
        else:
            if len(S) == 1:
                self.last = 0
                return (S[0], abs(S[0]))
            if self.n == 0:
                self.a1 = S[1] - S[0]
                self.run(S[0], S[0], self.a1)
            while self.n < len(S) - 1:
                na1 = S[self.n + 1] - S[self.n]
                self.run(S[self.n], self.a1, na1)
                self.a1 = na1
        value = self.A[0] / self.B[0]
        err = abs(value - self.last)
        self.last = value
        return (value, err)

    def update(self, X):
        """
        This routine applies the convergence acceleration to the list of individual terms.

        A = sum(a_k, k = 0..infinity)

        v, e = ...update([a_0, a_1,..., a_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
        if self.variant != 'v':
            if self.n == 0:
                self.s = X[0]
                self.run(self.s, X[0])
            while self.n < len(X):
                self.s += X[self.n]
                self.run(self.s, X[self.n])
        else:
            if len(X) == 1:
                self.last = 0
                return (X[0], abs(X[0]))
            if self.n == 0:
                self.s = X[0]
                self.run(self.s, X[0], X[1])
            while self.n < len(X) - 1:
                self.s += X[self.n]
                self.run(self.s, X[self.n], X[self.n + 1])
        value = self.A[0] / self.B[0]
        err = abs(value - self.last)
        self.last = value
        return (value, err)

    def step_psum(self, s):
        """
        This routine applies the convergence acceleration to the partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k, k = 0..n)

        v, e = ...step_psum(s_k)

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
        if self.variant != 'v':
            if self.n == 0:
                self.last_s = s
                self.run(s, s)
            else:
                self.run(s, s - self.last_s)
                self.last_s = s
        else:
            if isinstance(self.last_s, bool):
                self.last_s = s
                self.last_w = s
                self.last = 0
                return (s, abs(s))
            na1 = s - self.last_s
            self.run(self.last_s, self.last_w, na1)
            self.last_w = na1
            self.last_s = s
        value = self.A[0] / self.B[0]
        err = abs(value - self.last)
        self.last = value
        return (value, err)

    def step(self, x):
        """
        This routine applies the convergence acceleration to the individual terms.

        A = sum(a_k, k = 0..infinity)

        v, e = ...step(a_k)

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
        if self.variant != 'v':
            if self.n == 0:
                self.s = x
                self.run(self.s, x)
            else:
                self.s += x
                self.run(self.s, x)
        else:
            if isinstance(self.last_s, bool):
                self.last_s = x
                self.s = 0
                self.last = 0
                return (x, abs(x))
            self.s += self.last_s
            self.run(self.s, self.last_s, x)
            self.last_s = x
        value = self.A[0] / self.B[0]
        err = abs(value - self.last)
        self.last = value
        return (value, err)