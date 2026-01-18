from functools import wraps
from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn
from mpmath import mp, workprec
class BesselBase(Function):
    """
    Abstract base class for Bessel-type functions.

    This class is meant to reduce code duplication.
    All Bessel-type functions can 1) be differentiated, with the derivatives
    expressed in terms of similar functions, and 2) be rewritten in terms
    of other Bessel-type functions.

    Here, Bessel-type functions are assumed to have one complex parameter.

    To use this base class, define class attributes ``_a`` and ``_b`` such that
    ``2*F_n' = -_a*F_{n+1} + b*F_{n-1}``.

    """

    @property
    def order(self):
        """ The order of the Bessel-type function. """
        return self.args[0]

    @property
    def argument(self):
        """ The argument of the Bessel-type function. """
        return self.args[1]

    @classmethod
    def eval(cls, nu, z):
        return

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        return self._b / 2 * self.__class__(self.order - 1, self.argument) - self._a / 2 * self.__class__(self.order + 1, self.argument)

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return self.__class__(self.order.conjugate(), z.conjugate())

    def _eval_is_meromorphic(self, x, a):
        nu, z = (self.order, self.argument)
        if nu.has(x):
            return False
        if not z._eval_is_meromorphic(x, a):
            return None
        z0 = z.subs(x, a)
        if nu.is_integer:
            if isinstance(self, (besselj, besseli, hn1, hn2, jn, yn)) or not nu.is_zero:
                return fuzzy_not(z0.is_infinite)
        return fuzzy_not(fuzzy_or([z0.is_zero, z0.is_infinite]))

    def _eval_expand_func(self, **hints):
        nu, z, f = (self.order, self.argument, self.__class__)
        if nu.is_real:
            if (nu - 1).is_positive:
                return -self._a * self._b * f(nu - 2, z)._eval_expand_func() + 2 * self._a * (nu - 1) * f(nu - 1, z)._eval_expand_func() / z
            elif (nu + 1).is_negative:
                return 2 * self._b * (nu + 1) * f(nu + 1, z)._eval_expand_func() / z - self._a * self._b * f(nu + 2, z)._eval_expand_func()
        return self

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import besselsimp
        return besselsimp(self)