from sympy.assumptions import Q, ask, AppliedPredicate
from sympy.core import Basic, Symbol
from sympy.core.logic import _fuzzy_group
from sympy.core.numbers import NaN, Number
from sympy.logic.boolalg import (And, BooleanTrue, BooleanFalse, conjuncts,
from sympy.utilities.exceptions import sympy_deprecation_warning
from ..predicates.common import CommutativePredicate, IsTruePredicate
class AskHandler:
    """Base class that all Ask Handlers must inherit."""

    def __new__(cls, *args, **kwargs):
        sympy_deprecation_warning('\n            The AskHandler system is deprecated. The AskHandler class should\n            be replaced with the multipledispatch handler of Predicate\n            ', deprecated_since_version='1.8', active_deprecations_target='deprecated-askhandler')
        return super().__new__(cls, *args, **kwargs)