from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ExtendedNonPositivePredicate(Predicate):
    """
    Nonpositive extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonpositive(x))`` is true iff ``x`` is extended real and
    ``x`` is not positive.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonpositive(-1))
    True
    >>> ask(Q.extended_nonpositive(oo))
    False
    >>> ask(Q.extended_nonpositive(0))
    True
    >>> ask(Q.extended_nonpositive(I))
    False

    """
    name = 'extended_nonpositive'
    handler = Dispatcher('ExtendedNonPositiveHandler')