from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class PositiveInfinitePredicate(Predicate):
    """
    Positive infinity predicate.

    ``Q.positive_infinite(x)`` is true iff ``x`` is positive infinity ``oo``.
    """
    name = 'positive_infinite'
    handler = Dispatcher('PositiveInfiniteHandler')