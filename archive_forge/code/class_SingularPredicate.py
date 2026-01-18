from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class SingularPredicate(Predicate):
    """
    Singular matrix predicate.

    A matrix is singular iff the value of its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.singular(X), Q.invertible(X))
    False
    >>> ask(Q.singular(X), ~Q.invertible(X))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/SingularMatrix.html

    """
    name = 'singular'
    handler = Dispatcher('SingularHandler', doc="Predicate fore key 'singular'.")