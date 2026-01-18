from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
class hypot(Function):
    """
    Represents the hypotenuse function.

    Explanation
    ===========

    The hypotenuse function is provided by e.g. the math library
    in the C99 standard, hence one may want to represent the function
    symbolically when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.codegen.cfunctions import hypot
    >>> hypot(3, 4).evalf() == 5.0
    True
    >>> hypot(x, y)
    hypot(x, y)
    >>> hypot(x, y).diff(x)
    x/hypot(x, y)

    """
    nargs = 2

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex in (1, 2):
            return 2 * self.args[argindex - 1] / (_Two * self.func(*self.args))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        return _hypot(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _hypot(arg)
    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow