from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrices import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning
class AddWithLimits(ExprWithLimits):
    """Represents unevaluated oriented additions.
        Parent class for Integral and Sum.
    """
    __slots__ = ()

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.summations import Sum
        pre = _common_new(cls, function, *symbols, discrete=issubclass(cls, Sum), **assumptions)
        if isinstance(pre, tuple):
            function, limits, orientation = pre
        else:
            return pre
        obj = Expr.__new__(cls, **assumptions)
        arglist = [orientation * function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative
        return obj

    def _eval_adjoint(self):
        if all((x.is_real for x in flatten(self.limits))):
            return self.func(self.function.adjoint(), *self.limits)
        return None

    def _eval_conjugate(self):
        if all((x.is_real for x in flatten(self.limits))):
            return self.func(self.function.conjugate(), *self.limits)
        return None

    def _eval_transpose(self):
        if all((x.is_real for x in flatten(self.limits))):
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_factor(self, **hints):
        if 1 == len(self.limits):
            summand = self.function.factor(**hints)
            if summand.is_Mul:
                out = sift(summand.args, lambda w: w.is_commutative and (not set(self.variables) & w.free_symbols))
                return Mul(*out[True]) * self.func(Mul(*out[False]), *self.limits)
        else:
            summand = self.func(self.function, *self.limits[0:-1]).factor()
            if not summand.has(self.variables[-1]):
                return self.func(1, [self.limits[-1]]).doit() * summand
            elif isinstance(summand, Mul):
                return self.func(summand, self.limits[-1]).factor()
        return self

    def _eval_expand_basic(self, **hints):
        summand = self.function.expand(**hints)
        force = hints.get('force', False)
        if summand.is_Add and (force or (summand.is_commutative and self.has_finite_limits is not False)):
            return Add(*[self.func(i, *self.limits) for i in summand.args])
        elif isinstance(summand, MatrixBase):
            return summand.applyfunc(lambda x: self.func(x, *self.limits))
        elif summand != self.function:
            return self.func(summand, *self.limits)
        return self