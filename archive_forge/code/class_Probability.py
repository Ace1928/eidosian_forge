import itertools
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Not
from sympy.core.parameters import global_parameters
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.stats import variance, covariance
from sympy.stats.rv import (RandomSymbol, pspace, dependent,
class Probability(Expr):
    """
    Symbolic expression for the probability.

    Examples
    ========

    >>> from sympy.stats import Probability, Normal
    >>> from sympy import Integral
    >>> X = Normal("X", 0, 1)
    >>> prob = Probability(X > 1)
    >>> prob
    Probability(X > 1)

    Integral representation:

    >>> prob.rewrite(Integral)
    Integral(sqrt(2)*exp(-_z**2/2)/(2*sqrt(pi)), (_z, 1, oo))

    Evaluation of the integral:

    >>> prob.evaluate_integral()
    sqrt(2)*(-sqrt(2)*sqrt(pi)*erf(sqrt(2)/2) + sqrt(2)*sqrt(pi))/(4*sqrt(pi))
    """

    def __new__(cls, prob, condition=None, **kwargs):
        prob = _sympify(prob)
        if condition is None:
            obj = Expr.__new__(cls, prob)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, prob, condition)
        obj._condition = condition
        return obj

    def doit(self, **hints):
        condition = self.args[0]
        given_condition = self._condition
        numsamples = hints.get('numsamples', False)
        for_rewrite = not hints.get('for_rewrite', False)
        if isinstance(condition, Not):
            return S.One - self.func(condition.args[0], given_condition, evaluate=for_rewrite).doit(**hints)
        if condition.has(RandomIndexedSymbol):
            return pspace(condition).probability(condition, given_condition, evaluate=for_rewrite)
        if isinstance(given_condition, RandomSymbol):
            condrv = random_symbols(condition)
            if len(condrv) == 1 and condrv[0] == given_condition:
                from sympy.stats.frv_types import BernoulliDistribution
                return BernoulliDistribution(self.func(condition).doit(**hints), 0, 1)
            if any((dependent(rv, given_condition) for rv in condrv)):
                return Probability(condition, given_condition)
            else:
                return Probability(condition).doit()
        if given_condition is not None and (not isinstance(given_condition, (Relational, Boolean))):
            raise ValueError('%s is not a relational or combination of relationals' % given_condition)
        if given_condition == False or condition is S.false:
            return S.Zero
        if not isinstance(condition, (Relational, Boolean)):
            raise ValueError('%s is not a relational or combination of relationals' % condition)
        if condition is S.true:
            return S.One
        if numsamples:
            return sampling_P(condition, given_condition, numsamples=numsamples)
        if given_condition is not None:
            return Probability(given(condition, given_condition)).doit()
        if pspace(condition) == PSpace():
            return Probability(condition, given_condition)
        result = pspace(condition).probability(condition)
        if hasattr(result, 'doit') and for_rewrite:
            return result.doit()
        else:
            return result

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return self.func(arg, condition=condition).doit(for_rewrite=True)
    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()