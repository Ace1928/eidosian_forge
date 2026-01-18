from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ExtendedNonZeroPredicate(Predicate):
    """
    Nonzero extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonzero(x))`` is true iff ``x`` is extended real and
    ``x`` is not zero.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonzero(-1))
    True
    >>> ask(Q.extended_nonzero(oo))
    True
    >>> ask(Q.extended_nonzero(I))
    False

    """
    name = 'extended_nonzero'
    handler = Dispatcher('ExtendedNonZeroHandler')