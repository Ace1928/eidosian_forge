from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ExtendedPositivePredicate(Predicate):
    """
    Positive extended real number predicate.

    Explanation
    ===========

    ``Q.extended_positive(x)`` is true iff ``x`` is extended real and
    `x > 0`, that is if ``x`` is in the interval `(0, \\infty]`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_positive(1))
    True
    >>> ask(Q.extended_positive(oo))
    True
    >>> ask(Q.extended_positive(I))
    False

    """
    name = 'extended_positive'
    handler = Dispatcher('ExtendedPositiveHandler')