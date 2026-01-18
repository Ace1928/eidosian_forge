from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.polys.specialpolys import symmetric_poly
class sech(ReciprocalHyperbolicFunction):
    """
    ``sech(x)`` is the hyperbolic secant of ``x``.

    The hyperbolic secant function is $\\frac{2}{e^x + e^{-x}}$

    Examples
    ========

    >>> from sympy import sech
    >>> from sympy.abc import x
    >>> sech(x)
    sech(x)

    See Also
    ========

    sinh, cosh, tanh, coth, csch, asinh, acosh
    """
    _reciprocal_of = cosh
    _is_even = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -tanh(self.args[0]) * sech(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            return euler(n) / factorial(n) * x ** n

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1 / cos(I * arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(I * arg)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I / sinh(arg + I * pi / 2)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return 1 / cosh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return True