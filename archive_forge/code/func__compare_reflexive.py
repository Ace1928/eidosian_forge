from typing import Optional
from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not
def _compare_reflexive(self, lhs, rhs):
    if lhs is S.NaN or rhs is S.NaN:
        return None
    reflexive = self.is_reflexive
    if reflexive is None:
        pass
    elif reflexive and lhs == rhs:
        return True
    elif not reflexive and lhs == rhs:
        return False
    return None