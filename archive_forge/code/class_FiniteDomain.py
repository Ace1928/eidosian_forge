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
class FiniteDomain(RandomDomain):
    """
    A domain with discrete finite support

    Represented using a FiniteSet.
    """
    is_Finite = True

    @property
    def symbols(self):
        return FiniteSet((sym for sym, val in self.elements))

    @property
    def elements(self):
        return self.args[0]

    @property
    def dict(self):
        return FiniteSet(*[Dict(dict(el)) for el in self.elements])

    def __contains__(self, other):
        return other in self.elements

    def __iter__(self):
        return self.elements.__iter__()

    def as_boolean(self):
        return Or(*[And(*[Eq(sym, val) for sym, val in item]) for item in self])