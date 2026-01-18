from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
class log1p(Function):
    """
    Represents the natural logarithm of a number plus one.

    Explanation
    ===========

    The benefit of using ``log1p(x)`` over ``log(x + 1)``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log1p
    >>> from sympy import expand_log
    >>> '%.0e' % expand_log(log1p(1e-99)).evalf()
    '1e-99'
    >>> from math import log
    >>> log(1 + 1e-99)
    0.0
    >>> log1p(x).diff(x)
    1/(x + 1)

    See Also
    ========

    expm1
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One / (self.args[0] + S.One)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        return _log1p(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log1p(arg)
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    @classmethod
    def eval(cls, arg):
        if arg.is_Rational:
            return log(arg + S.One)
        elif not arg.is_Float:
            return log.eval(arg + S.One)
        elif arg.is_number:
            return log(Rational(arg) + S.One)

    def _eval_is_real(self):
        return (self.args[0] + S.One).is_nonnegative

    def _eval_is_finite(self):
        if (self.args[0] + S.One).is_zero:
            return False
        return self.args[0].is_finite

    def _eval_is_positive(self):
        return self.args[0].is_positive

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative