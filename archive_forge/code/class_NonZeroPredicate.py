from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class NonZeroPredicate(Predicate):
    """
    Nonzero real number predicate.

    Explanation
    ===========

    ``ask(Q.nonzero(x))`` is true iff ``x`` is real and ``x`` is not zero.  Note in
    particular that ``Q.nonzero(x)`` is false if ``x`` is not real.  Use
    ``~Q.zero(x)`` if you want the negation of being zero without any real
    assumptions.

    A few important facts about nonzero numbers:

    - ``Q.nonzero`` is logically equivalent to ``Q.positive | Q.negative``.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I, oo
    >>> x = symbols('x')
    >>> print(ask(Q.nonzero(x), ~Q.zero(x)))
    None
    >>> ask(Q.nonzero(x), Q.positive(x))
    True
    >>> ask(Q.nonzero(x), Q.zero(x))
    False
    >>> ask(Q.nonzero(0))
    False
    >>> ask(Q.nonzero(I))
    False
    >>> ask(~Q.zero(I))
    True
    >>> ask(Q.nonzero(oo))
    False

    """
    name = 'nonzero'
    handler = Dispatcher('NonZeroHandler', doc="Handler for key 'zero'. Test that an expression is not identically zero.")