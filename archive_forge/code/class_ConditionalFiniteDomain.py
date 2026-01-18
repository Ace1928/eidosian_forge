from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (I, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Intersection
from sympy.core.containers import Dict
from sympy.core.logic import Logic
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet
from sympy.stats.rv import (RandomDomain, ProductDomain, ConditionalDomain,
class ConditionalFiniteDomain(ConditionalDomain, ProductFiniteDomain):
    """
    A FiniteDomain that has been restricted by a condition

    Example: The possibilities of a die roll under the condition that the
    roll is even.
    """

    def __new__(cls, domain, condition):
        """
        Create a new instance of ConditionalFiniteDomain class
        """
        if condition is True:
            return domain
        cond = rv_subs(condition)
        return Basic.__new__(cls, domain, cond)

    def _test(self, elem):
        """
        Test the value. If value is boolean, return it. If value is equality
        relational (two objects are equal), return it with left-hand side
        being equal to right-hand side. Otherwise, raise ValueError exception.
        """
        val = self.condition.xreplace(dict(elem))
        if val in [True, False]:
            return val
        elif val.is_Equality:
            return val.lhs == val.rhs
        raise ValueError('Undecidable if %s' % str(val))

    def __contains__(self, other):
        return other in self.fulldomain and self._test(other)

    def __iter__(self):
        return (elem for elem in self.fulldomain if self._test(elem))

    @property
    def set(self):
        if isinstance(self.fulldomain, SingleFiniteDomain):
            return FiniteSet(*[elem for elem in self.fulldomain.set if frozenset(((self.fulldomain.symbol, elem),)) in self])
        else:
            raise NotImplementedError('Not implemented on multi-dimensional conditional domain')

    def as_boolean(self):
        return FiniteDomain.as_boolean(self)