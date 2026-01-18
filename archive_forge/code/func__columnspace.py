from .utilities import _iszero
def _columnspace(M, simplify=False):
    """Returns a list of vectors (Matrix objects) that span columnspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.columnspace()
    [Matrix([
    [ 1],
    [-2],
    [ 3]]), Matrix([
    [0],
    [0],
    [6]])]

    See Also
    ========

    nullspace
    rowspace
    """
    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)
    return [M.col(i) for i in pivots]