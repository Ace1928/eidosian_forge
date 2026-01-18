from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class InfinitePredicate(Predicate):
    """
    Infinite number predicate.

    ``Q.infinite(x)`` is true iff the absolute value of ``x`` is
    infinity.

    """
    name = 'infinite'
    handler = Dispatcher('InfiniteHandler', doc='Handler for Q.infinite key.')