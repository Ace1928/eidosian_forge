from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
class HyperRep(Function):
    """
    A base class for "hyper representation functions".

    This is used exclusively in ``hyperexpand()``, but fits more logically here.

    pFq is branched at 1 if p == q+1. For use with slater-expansion, we want
    define an "analytic continuation" to all polar numbers, which is
    continuous on circles and on the ray t*exp_polar(I*pi). Moreover, we want
    a "nice" expression for the various cases.

    This base class contains the core logic, concrete derived classes only
    supply the actual functions.

    """

    @classmethod
    def eval(cls, *args):
        newargs = tuple(map(unpolarify, args[:-1])) + args[-1:]
        if args != newargs:
            return cls(*newargs)

    @classmethod
    def _expr_small(cls, x):
        """ An expression for F(x) which holds for |x| < 1. """
        raise NotImplementedError

    @classmethod
    def _expr_small_minus(cls, x):
        """ An expression for F(-x) which holds for |x| < 1. """
        raise NotImplementedError

    @classmethod
    def _expr_big(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n)*x), |x| > 1. """
        raise NotImplementedError

    @classmethod
    def _expr_big_minus(cls, x, n):
        """ An expression for F(exp_polar(2*I*pi*n + pi*I)*x), |x| > 1. """
        raise NotImplementedError

    def _eval_rewrite_as_nonrep(self, *args, **kwargs):
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        minus = False
        newargs = self.args[:-1] + (x,)
        if not n.is_Integer:
            minus = True
            n -= S.Half
        newerargs = newargs + (n,)
        if minus:
            small = self._expr_small_minus(*newargs)
            big = self._expr_big_minus(*newerargs)
        else:
            small = self._expr_small(*newargs)
            big = self._expr_big(*newerargs)
        if big == small:
            return small
        return Piecewise((big, abs(x) > 1), (small, True))

    def _eval_rewrite_as_nonrepsmall(self, *args, **kwargs):
        x, n = self.args[-1].extract_branch_factor(allow_half=True)
        args = self.args[:-1] + (x,)
        if not n.is_Integer:
            return self._expr_small_minus(*args)
        return self._expr_small(*args)