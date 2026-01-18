from sympy.core import EulerGamma  # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, expand_mul
from sympy.core.numbers import I, pi, Rational
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg
class _erfs(Function):
    """
    Helper function to make the $\\mathrm{erf}(z)$ function
    tractable for the Gruntz algorithm.

    """

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point is S.Infinity:
            z = self.args[0]
            l = [1 / sqrt(pi) * factorial(2 * k) * (-S(4)) ** (-k) / factorial(k) * (1 / z) ** (2 * k + 1) for k in range(n)]
            o = Order(1 / z ** (2 * n + 1), x)
            return Add(*l)._eval_nseries(x, n, logx) + o
        t = point.extract_multiplicatively(I)
        if t is S.Infinity:
            z = self.args[0]
            l = [1 / sqrt(pi) * factorial(2 * k) * (-S(4)) ** (-k) / factorial(k) * (1 / z) ** (2 * k + 1) for k in range(n)]
            o = Order(1 / z ** (2 * n + 1), x)
            return Add(*l)._eval_nseries(x, n, logx) + o
        return super()._eval_aseries(n, args0, x, logx)

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -2 / sqrt(pi) + 2 * z * _erfs(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return (S.One - erf(z)) * exp(z ** 2)