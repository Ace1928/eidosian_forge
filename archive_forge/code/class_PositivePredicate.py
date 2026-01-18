from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class PositivePredicate(Predicate):
    """
    Positive real number predicate.

    Explanation
    ===========

    ``Q.positive(x)`` is true iff ``x`` is real and `x > 0`, that is if ``x``
    is in the interval `(0, \\infty)`.  In particular, infinity is not
    positive.

    A few important facts about positive numbers:

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.positive(x), Q.real(x) & ~Q.negative(x) & ~Q.zero(x))
    True
    >>> ask(Q.positive(1))
    True
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(~Q.positive(I))
    True

    """
    name = 'positive'
    handler = Dispatcher('PositiveHandler', doc="Handler for key 'positive'. Test that an expression is strictly greater than zero.")