from sympy.assumptions import Predicate, AppliedPredicate, Q
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.multipledispatch import Dispatcher
class CommutativePredicate(Predicate):
    """
    Commutative predicate.

    Explanation
    ===========

    ``ask(Q.commutative(x))`` is true iff ``x`` commutes with any other
    object with respect to multiplication operation.

    """
    name = 'commutative'
    handler = Dispatcher('CommutativeHandler', doc="Handler for key 'commutative'.")