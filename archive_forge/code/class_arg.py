from typing import Tuple as tTuple
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
class arg(Function):
    """
    Returns the argument (in radians) of a complex number. The argument is
    evaluated in consistent convention with ``atan2`` where the branch-cut is
    taken along the negative real axis and ``arg(z)`` is in the interval
    $(-\\pi,\\pi]$. For a positive number, the argument is always 0; the
    argument of a negative number is $\\pi$; and the argument of 0
    is undefined and returns ``nan``. So the ``arg`` function will never nest
    greater than 3 levels since at the 4th application, the result must be
    nan; for a real number, nan is returned on the 3rd application.

    Examples
    ========

    >>> from sympy import arg, I, sqrt, Dummy
    >>> from sympy.abc import x
    >>> arg(2.0)
    0
    >>> arg(I)
    pi/2
    >>> arg(sqrt(2) + I*sqrt(2))
    pi/4
    >>> arg(sqrt(3)/2 + I/2)
    pi/6
    >>> arg(4 + 3*I)
    atan(3/4)
    >>> arg(0.8 + 0.6*I)
    0.643501108793284
    >>> arg(arg(arg(arg(x))))
    nan
    >>> real = Dummy(real=True)
    >>> arg(arg(arg(real)))
    nan

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    value : Expr
        Returns arc tangent of arg measured in radians.

    """
    is_extended_real = True
    is_real = True
    is_finite = True
    _singularities = True

    @classmethod
    def eval(cls, arg):
        a = arg
        for i in range(3):
            if isinstance(a, cls):
                a = a.args[0]
            else:
                if i == 2 and a.is_extended_real:
                    return S.NaN
                break
        else:
            return S.NaN
        from sympy.functions.elementary.exponential import exp_polar
        if isinstance(arg, exp_polar):
            return periodic_argument(arg, oo)
        if not arg.is_Atom:
            c, arg_ = factor_terms(arg).as_coeff_Mul()
            if arg_.is_Mul:
                arg_ = Mul(*[a if sign(a) not in (-1, 1) else sign(a) for a in arg_.args])
            arg_ = sign(c) * arg_
        else:
            arg_ = arg
        if any((i.is_extended_positive is None for i in arg_.atoms(AppliedUndef))):
            return
        from sympy.functions.elementary.trigonometric import atan2
        x, y = arg_.as_real_imag()
        rv = atan2(y, x)
        if rv.is_number:
            return rv
        if arg_ != arg:
            return cls(arg_, evaluate=False)

    def _eval_derivative(self, t):
        x, y = self.args[0].as_real_imag()
        return (x * Derivative(y, t, evaluate=True) - y * Derivative(x, t, evaluate=True)) / (x ** 2 + y ** 2)

    def _eval_rewrite_as_atan2(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import atan2
        x, y = self.args[0].as_real_imag()
        return atan2(y, x)