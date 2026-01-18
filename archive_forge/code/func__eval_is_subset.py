from sympy.core.decorators import _sympifyit
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from .sets import Set, FiniteSet, SetKind
def _eval_is_subset(self, other):
    if isinstance(other, PowerSet):
        return self.arg.is_subset(other.arg)