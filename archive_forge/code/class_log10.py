from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
class log10(Function):
    """
    Represents the logarithm function with base ten.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log10
    >>> log10(100).evalf() == 2.0
    True
    >>> log10(x).diff(x)
    1/(x*log(10))

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
            return S.One / (log(_Ten) * self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            result = log.eval(arg, base=_Ten)
            if result.is_Atom:
                return result
        elif arg.is_Pow and arg.base == _Ten:
            return arg.exp

    def _eval_expand_func(self, **hints):
        return _log10(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log10(arg)
    _eval_rewrite_as_tractable = _eval_rewrite_as_log