from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class CompositePredicate(Predicate):
    """
    Composite number predicate.

    Explanation
    ===========

    ``ask(Q.composite(x))`` is true iff ``x`` is a positive integer and has
    at least one positive divisor other than ``1`` and the number itself.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> ask(Q.composite(0))
    False
    >>> ask(Q.composite(1))
    False
    >>> ask(Q.composite(2))
    False
    >>> ask(Q.composite(20))
    True

    """
    name = 'composite'
    handler = Dispatcher('CompositeHandler', doc="Handler for key 'composite'.")