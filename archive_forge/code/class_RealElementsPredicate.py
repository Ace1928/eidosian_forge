from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class RealElementsPredicate(Predicate):
    """
    Real elements matrix predicate.

    Explanation
    ===========

    ``Q.real_elements(x)`` is true iff all the elements of ``x``
    are real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.real(X[1, 2]), Q.real_elements(X))
    True

    """
    name = 'real_elements'
    handler = Dispatcher('RealElementsHandler', doc="Handler for key 'real_elements'.")