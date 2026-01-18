from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class FullRankPredicate(Predicate):
    """
    Fullrank matrix predicate.

    Explanation
    ===========

    ``Q.fullrank(x)`` is true iff ``x`` is a full rank matrix.
    A matrix is full rank if all rows and columns of the matrix
    are linearly independent. A square matrix is full rank iff
    its determinant is nonzero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.fullrank(X.T), Q.fullrank(X))
    True
    >>> ask(Q.fullrank(ZeroMatrix(3, 3)))
    False
    >>> ask(Q.fullrank(Identity(3)))
    True

    """
    name = 'fullrank'
    handler = Dispatcher('FullRankHandler', doc="Handler for key 'fullrank'.")