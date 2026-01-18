from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class InvertiblePredicate(Predicate):
    """
    Invertible matrix predicate.

    Explanation
    ===========

    ``Q.invertible(x)`` is true iff ``x`` is an invertible matrix.
    A square matrix is called invertible only if its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.invertible(X*Y), Q.invertible(X))
    False
    >>> ask(Q.invertible(X*Z), Q.invertible(X) & Q.invertible(Z))
    True
    >>> ask(Q.invertible(X), Q.fullrank(X) & Q.square(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Invertible_matrix

    """
    name = 'invertible'
    handler = Dispatcher('InvertibleHandler', doc='Handler for Q.invertible.')