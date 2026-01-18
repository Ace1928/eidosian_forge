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
class _eis(Function):
    """
    Helper function to make the $\\mathrm{Ei}(z)$ and $\\mathrm{li}(z)$
    functions tractable for the Gruntz algorithm.

    """

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[0] != S.Infinity:
            return super(_erfs, self)._eval_aseries(n, args0, x, logx)
        z = self.args[0]
        l = [factorial(k) * (1 / z) ** (k + 1) for k in range(n)]
        o = Order(1 / z ** (n + 1), x)
        return Add(*l)._eval_nseries(x, n, logx) + o

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return S.One / z - _eis(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return exp(-z) * Ei(z)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)