from sympy.core import S
from sympy.core.sympify import _sympify
from sympy.functions import KroneckerDelta
from .matexpr import MatrixExpr
from .special import ZeroMatrix, Identity, OneMatrix
class MatrixPermute(MatrixExpr):
    """Symbolic representation for permuting matrix rows or columns.

    Parameters
    ==========

    perm : Permutation, PermutationMatrix
        The permutation to use for permuting the matrix.
        The permutation can be resized to the suitable one,

    axis : 0 or 1
        The axis to permute alongside.
        If `0`, it will permute the matrix rows.
        If `1`, it will permute the matrix columns.

    Notes
    =====

    This follows the same notation used in
    :meth:`sympy.matrices.common.MatrixCommon.permute`.

    Examples
    ========

    >>> from sympy import Matrix, MatrixPermute
    >>> from sympy.combinatorics import Permutation

    Permuting the matrix rows:

    >>> p = Permutation(1, 2, 0)
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = MatrixPermute(A, p, axis=0)
    >>> B.as_explicit()
    Matrix([
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3]])

    Permuting the matrix columns:

    >>> B = MatrixPermute(A, p, axis=1)
    >>> B.as_explicit()
    Matrix([
    [2, 3, 1],
    [5, 6, 4],
    [8, 9, 7]])

    See Also
    ========

    sympy.matrices.common.MatrixCommon.permute
    """

    def __new__(cls, mat, perm, axis=S.Zero):
        from sympy.combinatorics.permutations import Permutation
        mat = _sympify(mat)
        if not mat.is_Matrix:
            raise ValueError('{} must be a SymPy matrix instance.'.format(perm))
        perm = _sympify(perm)
        if isinstance(perm, PermutationMatrix):
            perm = perm.args[0]
        if not isinstance(perm, Permutation):
            raise ValueError('{} must be a SymPy Permutation or a PermutationMatrix instance'.format(perm))
        axis = _sympify(axis)
        if axis not in (0, 1):
            raise ValueError('The axis must be 0 or 1.')
        mat_size = mat.shape[axis]
        if mat_size != perm.size:
            try:
                perm = perm.resize(mat_size)
            except ValueError:
                raise ValueError('Size does not match between the permutation {} and the matrix {} threaded over the axis {} and cannot be converted.'.format(perm, mat, axis))
        return super().__new__(cls, mat, perm, axis)

    def doit(self, deep=True, **hints):
        mat, perm, axis = self.args
        if deep:
            mat = mat.doit(deep=deep, **hints)
            perm = perm.doit(deep=deep, **hints)
        if perm.is_Identity:
            return mat
        if mat.is_Identity:
            if axis is S.Zero:
                return PermutationMatrix(perm)
            elif axis is S.One:
                return PermutationMatrix(perm ** (-1))
        if isinstance(mat, (ZeroMatrix, OneMatrix)):
            return mat
        if isinstance(mat, MatrixPermute) and mat.args[2] == axis:
            return MatrixPermute(mat.args[0], perm * mat.args[1], axis)
        return self

    @property
    def shape(self):
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        mat, perm, axis = self.args
        if axis == 0:
            return mat[perm.apply(i), j]
        elif axis == 1:
            return mat[i, perm.apply(j)]

    def _eval_rewrite_as_MatMul(self, *args, **kwargs):
        from .matmul import MatMul
        mat, perm, axis = self.args
        deep = kwargs.get('deep', True)
        if deep:
            mat = mat.rewrite(MatMul)
        if axis == 0:
            return MatMul(PermutationMatrix(perm), mat)
        elif axis == 1:
            return MatMul(mat, PermutationMatrix(perm ** (-1)))