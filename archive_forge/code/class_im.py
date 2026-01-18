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
class im(Function):
    """
    Returns imaginary part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly more
    complicated expressions. If completely simplified result is needed then
    use ``Basic.as_real_imag()`` or perform complex expansion on instance of
    this function.

    Examples
    ========

    >>> from sympy import re, im, E, I
    >>> from sympy.abc import x, y
    >>> im(2*E)
    0
    >>> im(2*I + 17)
    2
    >>> im(x*I)
    re(x)
    >>> im(re(x) + y)
    im(y)
    >>> im(2 + 3*I)
    3

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Imaginary part of expression.

    See Also
    ========

    re
    """
    args: tTuple[Expr]
    is_extended_real = True
    unbranched = True
    _singularities = True

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return S.Zero
        elif arg.is_imaginary or (I * arg).is_extended_real:
            return -I * arg
        elif arg.is_Matrix:
            return arg.as_real_imag()[1]
        elif arg.is_Function and isinstance(arg, conjugate):
            return -im(arg.args[0])
        else:
            included, reverted, excluded = ([], [], [])
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)
                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                    else:
                        excluded.append(coeff)
                elif term.has(I) or not term.is_extended_real:
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[1])
                    else:
                        included.append(term)
            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])
                return cls(a) + re(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Return the imaginary part with a zero real part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        if x.is_extended_real or self.args[0].is_extended_real:
            return im(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I * re(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_re(self, arg, **kwargs):
        return -I * (self.args[0] - re(self.args[0]))

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_finite:
            return True