from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class NonPositivePredicate(Predicate):
    """
    Nonpositive real number predicate.

    Explanation
    ===========

    ``ask(Q.nonpositive(x))`` is true iff ``x`` belongs to the set of
    negative numbers including zero.

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I

    >>> ask(Q.nonpositive(-1))
    True
    >>> ask(Q.nonpositive(0))
    True
    >>> ask(Q.nonpositive(1))
    False
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(Q.nonpositive(-I))
    False

    """
    name = 'nonpositive'
    handler = Dispatcher('NonPositiveHandler', doc="Handler for key 'nonpositive'.")