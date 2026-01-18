from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class RationalPredicate(Predicate):
    """
    Rational number predicate.

    Explanation
    ===========

    ``Q.rational(x)`` is true iff ``x`` belongs to the set of
    rational numbers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S
    >>> ask(Q.rational(0))
    True
    >>> ask(Q.rational(S(1)/2))
    True
    >>> ask(Q.rational(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rational_number

    """
    name = 'rational'
    handler = Dispatcher('RationalHandler', doc='Handler for Q.rational.\n\nTest that an expression belongs to the field of rational numbers.')