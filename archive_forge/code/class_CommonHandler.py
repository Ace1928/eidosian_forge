from sympy.assumptions import Q, ask, AppliedPredicate
from sympy.core import Basic, Symbol
from sympy.core.logic import _fuzzy_group
from sympy.core.numbers import NaN, Number
from sympy.logic.boolalg import (And, BooleanTrue, BooleanFalse, conjuncts,
from sympy.utilities.exceptions import sympy_deprecation_warning
from ..predicates.common import CommutativePredicate, IsTruePredicate
class CommonHandler(AskHandler):
    """Defines some useful methods common to most Handlers. """

    @staticmethod
    def AlwaysTrue(expr, assumptions):
        return True

    @staticmethod
    def AlwaysFalse(expr, assumptions):
        return False

    @staticmethod
    def AlwaysNone(expr, assumptions):
        return None
    NaN = AlwaysFalse