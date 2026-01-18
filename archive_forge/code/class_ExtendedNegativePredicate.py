from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ExtendedNegativePredicate(Predicate):
    """
    Negative extended real number predicate.

    Explanation
    ===========

    ``Q.extended_negative(x)`` is true iff ``x`` is extended real and
    `x < 0`, that is if ``x`` is in the interval `[-\\infty, 0)`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_negative(-1))
    True
    >>> ask(Q.extended_negative(-oo))
    True
    >>> ask(Q.extended_negative(-I))
    False

    """
    name = 'extended_negative'
    handler = Dispatcher('ExtendedNegativeHandler')