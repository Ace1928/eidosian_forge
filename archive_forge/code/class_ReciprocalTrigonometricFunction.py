from typing import Tuple as tTuple, Union as tUnion
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.relational import Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.combinatorial.numbers import bernoulli, euler
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary._trigonometric_special import (
from sympy.logic.boolalg import And
from sympy.ntheory import factorint
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.iterables import numbered_symbols
class ReciprocalTrigonometricFunction(TrigonometricFunction):
    """Base class for reciprocal functions of trigonometric functions. """
    _reciprocal_of = None
    _singularities = (S.ComplexInfinity,)
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        if arg.could_extract_minus_sign():
            if cls._is_even:
                return cls(-arg)
            if cls._is_odd:
                return -cls(-arg)
        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None and (not (2 * pi_coeff).is_integer) and pi_coeff.is_Rational:
            q = pi_coeff.q
            p = pi_coeff.p % (2 * q)
            if p > q:
                narg = (pi_coeff - 1) * pi
                return -cls(narg)
            if 2 * p > q:
                narg = (1 - pi_coeff) * pi
                if cls._is_odd:
                    return cls(narg)
                elif cls._is_even:
                    return -cls(narg)
        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]
        t = cls._reciprocal_of.eval(arg)
        if t is None:
            return t
        elif any((isinstance(i, cos) for i in (t, -t))):
            return (1 / t).rewrite(sec)
        elif any((isinstance(i, sin) for i in (t, -t))):
            return (1 / t).rewrite(csc)
        else:
            return 1 / t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1 / t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1 / t

    def _period(self, symbol):
        f = expand_mul(self.args[0])
        return self._reciprocal_of(f).period(symbol)

    def fdiff(self, argindex=1):
        return -self._calculate_reciprocal('fdiff', argindex) / self ** 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_exp', arg)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_Pow', arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_sin', arg)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_cos', arg)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_tan', arg)

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_pow', arg)

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        return self._rewrite_reciprocal('_eval_rewrite_as_sqrt', arg)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        return (1 / self._reciprocal_of(self.args[0])).as_real_imag(deep, **hints)

    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal('_eval_expand_trig', **hints)

    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0])._eval_is_extended_real()

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        return (1 / self._reciprocal_of(self.args[0]))._eval_as_leading_term(x)

    def _eval_is_finite(self):
        return (1 / self._reciprocal_of(self.args[0])).is_finite

    def _eval_nseries(self, x, n, logx, cdir=0):
        return (1 / self._reciprocal_of(self.args[0]))._eval_nseries(x, n, logx)