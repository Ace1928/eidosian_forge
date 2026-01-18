from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
class LambertW(Function):
    """
    The Lambert W function $W(z)$ is defined as the inverse
    function of $w \\exp(w)$ [1]_.

    Explanation
    ===========

    In other words, the value of $W(z)$ is such that $z = W(z) \\exp(W(z))$
    for any complex number $z$.  The Lambert W function is a multivalued
    function with infinitely many branches $W_k(z)$, indexed by
    $k \\in \\mathbb{Z}$.  Each branch gives a different solution $w$
    of the equation $z = w \\exp(w)$.

    The Lambert W function has two partially real branches: the
    principal branch ($k = 0$) is real for real $z > -1/e$, and the
    $k = -1$ branch is real for $-1/e < z < 0$. All branches except
    $k = 0$ have a logarithmic singularity at $z = 0$.

    Examples
    ========

    >>> from sympy import LambertW
    >>> LambertW(1.2)
    0.635564016364870
    >>> LambertW(1.2, -1).n()
    -1.34747534407696 - 4.41624341514535*I
    >>> LambertW(-1).is_real
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lambert_W_function
    """
    _singularities = (-Pow(S.Exp1, -1, evaluate=False), S.ComplexInfinity)

    @classmethod
    def eval(cls, x, k=None):
        if k == S.Zero:
            return cls(x)
        elif k is None:
            k = S.Zero
        if k.is_zero:
            if x.is_zero:
                return S.Zero
            if x is S.Exp1:
                return S.One
            if x == -1 / S.Exp1:
                return S.NegativeOne
            if x == -log(2) / 2:
                return -log(2)
            if x == 2 * log(2):
                return log(2)
            if x == -pi / 2:
                return I * pi / 2
            if x == exp(1 + S.Exp1):
                return S.Exp1
            if x is S.Infinity:
                return S.Infinity
            if x.is_zero:
                return S.Zero
        if fuzzy_not(k.is_zero):
            if x.is_zero:
                return S.NegativeInfinity
        if k is S.NegativeOne:
            if x == -pi / 2:
                return -I * pi / 2
            elif x == -1 / S.Exp1:
                return S.NegativeOne
            elif x == -2 * exp(-2):
                return -Integer(2)

    def fdiff(self, argindex=1):
        """
        Return the first derivative of this function.
        """
        x = self.args[0]
        if len(self.args) == 1:
            if argindex == 1:
                return LambertW(x) / (x * (1 + LambertW(x)))
        else:
            k = self.args[1]
            if argindex == 1:
                return LambertW(x, k) / (x * (1 + LambertW(x, k)))
        raise ArgumentIndexError(self, argindex)

    def _eval_is_extended_real(self):
        x = self.args[0]
        if len(self.args) == 1:
            k = S.Zero
        else:
            k = self.args[1]
        if k.is_zero:
            if (x + 1 / S.Exp1).is_positive:
                return True
            elif (x + 1 / S.Exp1).is_nonpositive:
                return False
        elif (k + 1).is_zero:
            if x.is_negative and (x + 1 / S.Exp1).is_positive:
                return True
            elif x.is_nonpositive or (x + 1 / S.Exp1).is_nonnegative:
                return False
        elif fuzzy_not(k.is_zero) and fuzzy_not((k + 1).is_zero):
            if x.is_extended_real:
                return False

    def _eval_is_finite(self):
        return self.args[0].is_finite

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
        else:
            return s.is_algebraic

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if len(self.args) == 1:
            arg = self.args[0]
            arg0 = arg.subs(x, 0).cancel()
            if not arg0.is_zero:
                return self.func(arg0)
            return arg.as_leading_term(x)

    def _eval_nseries(self, x, n, logx, cdir=0):
        if len(self.args) == 1:
            from sympy.functions.elementary.integers import ceiling
            from sympy.series.order import Order
            arg = self.args[0].nseries(x, n=n, logx=logx)
            lt = arg.as_leading_term(x, logx=logx)
            lte = 1
            if lt.is_Pow:
                lte = lt.exp
            if ceiling(n / lte) >= 1:
                s = Add(*[(-S.One) ** (k - 1) * Integer(k) ** (k - 2) / factorial(k - 1) * arg ** k for k in range(1, ceiling(n / lte))])
                s = expand_multinomial(s)
            else:
                s = S.Zero
            return s + Order(x ** n, x)
        return super()._eval_nseries(x, n, logx)

    def _eval_is_zero(self):
        x = self.args[0]
        if len(self.args) == 1:
            return x.is_zero
        else:
            return fuzzy_and([x.is_zero, self.args[1].is_zero])