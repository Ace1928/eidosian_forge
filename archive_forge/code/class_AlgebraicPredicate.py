from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class AlgebraicPredicate(Predicate):
    """
    Algebraic number predicate.

    Explanation
    ===========

    ``Q.algebraic(x)`` is true iff ``x`` belongs to the set of
    algebraic numbers. ``x`` is algebraic if there is some polynomial
    in ``p(x)\\in \\mathbb\\{Q\\}[x]`` such that ``p(x) = 0``.

    Examples
    ========

    >>> from sympy import ask, Q, sqrt, I, pi
    >>> ask(Q.algebraic(sqrt(2)))
    True
    >>> ask(Q.algebraic(I))
    True
    >>> ask(Q.algebraic(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Algebraic_number

    """
    name = 'algebraic'
    AlgebraicHandler = Dispatcher('AlgebraicHandler', doc='Handler for Q.algebraic key.')