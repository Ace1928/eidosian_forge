from sympy.assumptions import Q, ask, AppliedPredicate
from sympy.core import Basic, Symbol
from sympy.core.logic import _fuzzy_group
from sympy.core.numbers import NaN, Number
from sympy.logic.boolalg import (And, BooleanTrue, BooleanFalse, conjuncts,
from sympy.utilities.exceptions import sympy_deprecation_warning
from ..predicates.common import CommutativePredicate, IsTruePredicate
@staticmethod
def AlwaysTrue(expr, assumptions):
    return True