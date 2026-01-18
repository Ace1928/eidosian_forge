from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class UnitTriangularPredicate(Predicate):
    """
    Unit triangular matrix predicate.

    Explanation
    ===========

    A unit triangular matrix is a triangular matrix with 1s
    on the diagonal.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.unit_triangular(X))
    True

    """
    name = 'unit_triangular'
    handler = Dispatcher('UnitTriangularHandler', doc="Predicate fore key 'unit_triangular'.")