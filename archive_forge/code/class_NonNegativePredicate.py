from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class NonNegativePredicate(Predicate):
    """
    Nonnegative real number predicate.

    Explanation
    ===========

    ``ask(Q.nonnegative(x))`` is true iff ``x`` belongs to the set of
    positive numbers including zero.

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.nonnegative(1))
    True
    >>> ask(Q.nonnegative(0))
    True
    >>> ask(Q.nonnegative(-1))
    False
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(Q.nonnegative(-I))
    False

    """
    name = 'nonnegative'
    handler = Dispatcher('NonNegativeHandler', doc='Handler for Q.nonnegative.')