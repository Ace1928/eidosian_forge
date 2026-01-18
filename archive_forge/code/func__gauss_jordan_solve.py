from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols
from .common import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero
def _gauss_jordan_solve(M, B, freevar=False):
    """
    Solves ``Ax = B`` using Gauss Jordan elimination.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned. If infinite solutions exist, it will
    be returned parametrically. If no solutions exist, It will throw
    ValueError.

    Parameters
    ==========

    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.

    freevar : boolean, optional
        Flag, when set to `True` will return the indices of the free
        variables in the solutions (column Matrix), for a system that is
        undetermined (e.g. A has more columns than rows), for which
        infinite solutions are possible, in terms of arbitrary
        values of free variables. Default `False`.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    params : Matrix
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        parameters. These arbitrary parameters are returned as params
        Matrix.

    free_var_index : List, optional
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        values of free variables. Then the indices of the free variables
        in the solutions (column Matrix) are returned by free_var_index,
        if the flag `freevar` is set to `True`.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> taus_zeroes = { tau:0 for tau in params }
    >>> sol_unique = sol.xreplace(taus_zeroes)
    >>> sol_unique
        Matrix([
    [2],
    [0],
    [5],
    [0]])


    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> B = Matrix([3, 6, 9])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-1],
    [ 2],
    [ 0]])
    >>> params
    Matrix(0, 1, [])

    >>> A = Matrix([[2, -7], [-1, 4]])
    >>> B = Matrix([[-21, 3], [12, -2]])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [0, -2],
    [3, -1]])
    >>> params
    Matrix(0, 2, [])


    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params, freevars = A.gauss_jordan_solve(B, freevar=True)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> freevars
    [1, 3]


    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_elimination

    """
    from sympy.matrices import Matrix, zeros
    cls = M.__class__
    aug = M.hstack(M.copy(), B.copy())
    B_cols = B.cols
    row, col = aug[:, :-B_cols].shape
    A, pivots = aug.rref(simplify=True)
    A, v = (A[:, :-B_cols], A[:, -B_cols:])
    pivots = list(filter(lambda p: p < col, pivots))
    rank = len(pivots)
    free_var_index = [c for c in range(A.cols) if c not in pivots]
    permutation = Matrix(pivots + free_var_index).T
    if not v[rank:, :].is_zero_matrix:
        raise ValueError('Linear system has no solution')
    name = uniquely_named_symbol('tau', aug, compare=lambda i: str(i).rstrip('1234567890'), modify=lambda s: '_' + s).name
    gen = numbered_symbols(name)
    tau = Matrix([next(gen) for k in range((col - rank) * B_cols)]).reshape(col - rank, B_cols)
    V = A[:rank, free_var_index]
    vt = v[:rank, :]
    free_sol = tau.vstack(vt - V * tau, tau)
    sol = zeros(col, B_cols)
    for k in range(col):
        sol[permutation[k], :] = free_sol[k, :]
    sol, tau = (cls(sol), cls(tau))
    if freevar:
        return (sol, tau, free_var_index)
    else:
        return (sol, tau)