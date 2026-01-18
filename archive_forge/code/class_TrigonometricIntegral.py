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
class TrigonometricIntegral(Function):
    """ Base class for trigonometric integrals. """

    @classmethod
    def eval(cls, z):
        if z is S.Zero:
            return cls._atzero
        elif z is S.Infinity:
            return cls._atinf()
        elif z is S.NegativeInfinity:
            return cls._atneginf()
        if z.is_zero:
            return cls._atzero
        nz = z.extract_multiplicatively(polar_lift(I))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(I)
        if nz is not None:
            return cls._Ifactor(nz, 1)
        nz = z.extract_multiplicatively(polar_lift(-I))
        if nz is not None:
            return cls._Ifactor(nz, -1)
        nz = z.extract_multiplicatively(polar_lift(-1))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(-1)
        if nz is not None:
            return cls._minusfactor(nz)
        nz, n = z.extract_branch_factor()
        if n == 0 and nz == z:
            return
        return 2 * pi * I * n * cls._trigfunc(0) + cls(nz)

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return self._trigfunc(arg) / arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return self._eval_rewrite_as_expint(z).rewrite(Ei)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return self._eval_rewrite_as_expint(z).rewrite(uppergamma)

    def _eval_nseries(self, x, n, logx, cdir=0):
        n += 1
        if self.args[0].subs(x, 0) != 0:
            return super()._eval_nseries(x, n, logx)
        baseseries = self._trigfunc(x)._eval_nseries(x, n, logx)
        if self._trigfunc(0) != 0:
            baseseries -= 1
        baseseries = baseseries.replace(Pow, lambda t, n: t ** n / n, simultaneous=False)
        if self._trigfunc(0) != 0:
            baseseries += EulerGamma + log(x)
        return baseseries.subs(x, self.args[0])._eval_nseries(x, n, logx)