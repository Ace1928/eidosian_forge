from .utilities import _iszero
def _nullspace(M, simplify=False, iszerofunc=_iszero):
    """Returns list of vectors (Matrix objects) that span nullspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.nullspace()
    [Matrix([
    [-3],
    [ 1],
    [ 0]])]

    See Also
    ========

    columnspace
    rowspace
    """
    reduced, pivots = M.rref(iszerofunc=iszerofunc, simplify=simplify)
    free_vars = [i for i in range(M.cols) if i not in pivots]
    basis = []
    for free_var in free_vars:
        vec = [M.zero] * M.cols
        vec[free_var] = M.one
        for piv_row, piv_col in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]
        basis.append(vec)
    return [M._new(M.cols, 1, b) for b in basis]