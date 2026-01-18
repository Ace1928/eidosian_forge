from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class IrrationalPredicate(Predicate):
    """
    Irrational number predicate.

    Explanation
    ===========

    ``Q.irrational(x)`` is true iff ``x``  is any real number that
    cannot be expressed as a ratio of integers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S, I
    >>> ask(Q.irrational(0))
    False
    >>> ask(Q.irrational(S(1)/2))
    False
    >>> ask(Q.irrational(pi))
    True
    >>> ask(Q.irrational(I))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Irrational_number

    """
    name = 'irrational'
    handler = Dispatcher('IrrationalHandler', doc='Handler for Q.irrational.\n\nTest that an expression is irrational numbers.')