from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class UnitaryPredicate(Predicate):
    """
    Unitary matrix predicate.

    Explanation
    ===========

    ``Q.unitary(x)`` is true iff ``x`` is a unitary matrix.
    Unitary matrix is an analogue to orthogonal matrix. A square
    matrix ``M`` with complex elements is unitary if :math:``M^TM = MM^T= I``
    where :math:``M^T`` is the conjugate transpose matrix of ``M``.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.unitary(Y))
    False
    >>> ask(Q.unitary(X*Z*X), Q.unitary(X) & Q.unitary(Z))
    True
    >>> ask(Q.unitary(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_matrix

    """
    name = 'unitary'
    handler = Dispatcher('UnitaryHandler', doc="Handler for key 'unitary'.")