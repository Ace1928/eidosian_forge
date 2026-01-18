from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
class exp2(Function):
    """
    Represents the exponential function with base two.

    Explanation
    ===========

    The benefit of using ``exp2(x)`` over ``2**x``
    is that the latter is not as efficient under finite precision
    arithmetic.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import exp2
    >>> exp2(2).evalf() == 4.0
    True
    >>> exp2(x).diff(x)
    log(2)*exp2(x)

    See Also
    ========

    log2
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return self * log(_Two)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _exp2(arg)
    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow

    def _eval_expand_func(self, **hints):
        return _exp2(*self.args)

    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            return _exp2(arg)