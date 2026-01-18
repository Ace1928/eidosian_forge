from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class SquarePredicate(Predicate):
    """
    Square matrix predicate.

    Explanation
    ===========

    ``Q.square(x)`` is true iff ``x`` is a square matrix. A square matrix
    is a matrix with the same number of rows and columns.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('X', 2, 3)
    >>> ask(Q.square(X))
    True
    >>> ask(Q.square(Y))
    False
    >>> ask(Q.square(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.square(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_matrix

    """
    name = 'square'
    handler = Dispatcher('SquareHandler', doc='Handler for Q.square.')