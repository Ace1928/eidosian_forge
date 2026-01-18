from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ExtendedRealPredicate(Predicate):
    """
    Extended real predicate.

    Explanation
    ===========

    ``Q.extended_real(x)`` is true iff ``x`` is a real number or
    `\\{-\\infty, \\infty\\}`.

    See documentation of ``Q.real`` for more information about related
    facts.

    Examples
    ========

    >>> from sympy import ask, Q, oo, I
    >>> ask(Q.extended_real(1))
    True
    >>> ask(Q.extended_real(I))
    False
    >>> ask(Q.extended_real(oo))
    True

    """
    name = 'extended_real'
    handler = Dispatcher('ExtendedRealHandler', doc='Handler for Q.extended_real.\n\nTest that an expression belongs to the field of extended real\nnumbers, that is real numbers union {Infinity, -Infinity}.')