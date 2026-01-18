from sympy.assumptions.ask import (Q, ask)
from sympy.core import Basic, Add, Mul, S
from sympy.core.sympify import _sympify
from sympy.functions import adjoint
from sympy.functions.elementary.complexes import re, im
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities.iterables import is_sequence, sift
from sympy.utilities.misc import filldedent
from sympy.matrices import Matrix, ShapeError
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.matrices.expressions.trace import trace
from sympy.matrices.expressions.transpose import Transpose, transpose
class BlockMatrix(MatrixExpr):
    """A BlockMatrix is a Matrix comprised of other matrices.

    The submatrices are stored in a SymPy Matrix object but accessed as part of
    a Matrix Expression

    >>> from sympy import (MatrixSymbol, BlockMatrix, symbols,
    ...     Identity, ZeroMatrix, block_collapse)
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])

    Some matrices might be comprised of rows of blocks with
    the matrices in each row having the same height and the
    rows all having the same total number of columns but
    not having the same number of columns for each matrix
    in each row. In this case, the matrix is not a block
    matrix and should be instantiated by Matrix.

    >>> from sympy import ones, Matrix
    >>> dat = [
    ... [ones(3,2), ones(3,3)*2],
    ... [ones(2,3)*3, ones(2,2)*4]]
    ...
    >>> BlockMatrix(dat)
    Traceback (most recent call last):
    ...
    ValueError:
    Although this matrix is comprised of blocks, the blocks do not fill
    the matrix in a size-symmetric fashion. To create a full matrix from
    these arguments, pass them directly to Matrix.
    >>> Matrix(dat)
    Matrix([
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4]])

    See Also
    ========
    sympy.matrices.matrices.MatrixBase.irregular
    """

    def __new__(cls, *args, **kwargs):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        isMat = lambda i: getattr(i, 'is_Matrix', False)
        if len(args) != 1 or not is_sequence(args[0]) or len({isMat(r) for r in args[0]}) != 1:
            raise ValueError(filldedent('\n                expecting a sequence of 1 or more rows\n                containing Matrices.'))
        rows = args[0] if args else []
        if not isMat(rows):
            if rows and isMat(rows[0]):
                rows = [rows]
            blocky = ok = len({len(r) for r in rows}) == 1
            if ok:
                for r in rows:
                    ok = len({i.rows for i in r}) == 1
                    if not ok:
                        break
                blocky = ok
                if ok:
                    for c in range(len(rows[0])):
                        ok = len({rows[i][c].cols for i in range(len(rows))}) == 1
                        if not ok:
                            break
            if not ok:
                ok = len({sum([i.cols for i in r]) for r in rows}) == 1
                if blocky and ok:
                    raise ValueError(filldedent('\n                        Although this matrix is comprised of blocks,\n                        the blocks do not fill the matrix in a\n                        size-symmetric fashion. To create a full matrix\n                        from these arguments, pass them directly to\n                        Matrix.'))
                raise ValueError(filldedent("\n                    When there are not the same number of rows in each\n                    row's matrices or there are not the same number of\n                    total columns in each row, the matrix is not a\n                    block matrix. If this matrix is known to consist of\n                    blocks fully filling a 2-D space then see\n                    Matrix.irregular."))
        mat = ImmutableDenseMatrix(rows, evaluate=False)
        obj = Basic.__new__(cls, mat)
        return obj

    @property
    def shape(self):
        numrows = numcols = 0
        M = self.blocks
        for i in range(M.shape[0]):
            numrows += M[i, 0].shape[0]
        for i in range(M.shape[1]):
            numcols += M[0, i].shape[1]
        return (numrows, numcols)

    @property
    def blockshape(self):
        return self.blocks.shape

    @property
    def blocks(self):
        return self.args[0]

    @property
    def rowblocksizes(self):
        return [self.blocks[i, 0].rows for i in range(self.blockshape[0])]

    @property
    def colblocksizes(self):
        return [self.blocks[0, i].cols for i in range(self.blockshape[1])]

    def structurally_equal(self, other):
        return isinstance(other, BlockMatrix) and self.shape == other.shape and (self.blockshape == other.blockshape) and (self.rowblocksizes == other.rowblocksizes) and (self.colblocksizes == other.colblocksizes)

    def _blockmul(self, other):
        if isinstance(other, BlockMatrix) and self.colblocksizes == other.rowblocksizes:
            return BlockMatrix(self.blocks * other.blocks)
        return self * other

    def _blockadd(self, other):
        if isinstance(other, BlockMatrix) and self.structurally_equal(other):
            return BlockMatrix(self.blocks + other.blocks)
        return self + other

    def _eval_transpose(self):
        matrices = [transpose(matrix) for matrix in self.blocks]
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        M = M.transpose()
        return BlockMatrix(M)

    def _eval_adjoint(self):
        matrices = [adjoint(matrix) for matrix in self.blocks]
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        M = M.transpose()
        return BlockMatrix(M)

    def _eval_trace(self):
        if self.rowblocksizes == self.colblocksizes:
            return Add(*[trace(self.blocks[i, i]) for i in range(self.blockshape[0])])
        raise NotImplementedError("Can't perform trace of irregular blockshape")

    def _eval_determinant(self):
        if self.blockshape == (1, 1):
            return det(self.blocks[0, 0])
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            if ask(Q.invertible(A)):
                return det(A) * det(D - C * A.I * B)
            elif ask(Q.invertible(D)):
                return det(D) * det(A - B * D.I * C)
        return Determinant(self)

    def _eval_as_real_imag(self):
        real_matrices = [re(matrix) for matrix in self.blocks]
        real_matrices = Matrix(self.blockshape[0], self.blockshape[1], real_matrices)
        im_matrices = [im(matrix) for matrix in self.blocks]
        im_matrices = Matrix(self.blockshape[0], self.blockshape[1], im_matrices)
        return (BlockMatrix(real_matrices), BlockMatrix(im_matrices))

    def transpose(self):
        """Return transpose of matrix.

        Examples
        ========

        >>> from sympy import MatrixSymbol, BlockMatrix, ZeroMatrix
        >>> from sympy.abc import m, n
        >>> X = MatrixSymbol('X', n, n)
        >>> Y = MatrixSymbol('Y', m, m)
        >>> Z = MatrixSymbol('Z', n, m)
        >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
        >>> B.transpose()
        Matrix([
        [X.T,  0],
        [Z.T, Y.T]])
        >>> _.transpose()
        Matrix([
        [X, Z],
        [0, Y]])
        """
        return self._eval_transpose()

    def schur(self, mat='A', generalized=False):
        """Return the Schur Complement of the 2x2 BlockMatrix

        Parameters
        ==========

        mat : String, optional
            The matrix with respect to which the
            Schur Complement is calculated. 'A' is
            used by default

        generalized : bool, optional
            If True, returns the generalized Schur
            Component which uses Moore-Penrose Inverse

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])

        The default Schur Complement is evaluated with "A"

        >>> X.schur()
        -C*A**(-1)*B + D
        >>> X.schur('D')
        A - B*D**(-1)*C

        Schur complement with non-invertible matrices is not
        defined. Instead, the generalized Schur complement can
        be calculated which uses the Moore-Penrose Inverse. To
        achieve this, `generalized` must be set to `True`

        >>> X.schur('B', generalized=True)
        C - D*(B.T*B)**(-1)*B.T*A
        >>> X.schur('C', generalized=True)
        -A*(C.T*C)**(-1)*C.T*D + B

        Returns
        =======

        M : Matrix
            The Schur Complement Matrix

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If given matrix is non-invertible

        References
        ==========

        .. [1] Wikipedia Article on Schur Component : https://en.wikipedia.org/wiki/Schur_complement

        See Also
        ========

        sympy.matrices.matrices.MatrixBase.pinv
        """
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            d = {'A': A, 'B': B, 'C': C, 'D': D}
            try:
                inv = (d[mat].T * d[mat]).inv() * d[mat].T if generalized else d[mat].inv()
                if mat == 'A':
                    return D - C * inv * B
                elif mat == 'B':
                    return C - D * inv * A
                elif mat == 'C':
                    return B - A * inv * D
                elif mat == 'D':
                    return A - B * inv * C
                return self
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('The given matrix is not invertible. Please set generalized=True             to compute the generalized Schur Complement which uses Moore-Penrose Inverse')
        else:
            raise ShapeError('Schur Complement can only be calculated for 2x2 block matrices')

    def LDUdecomposition(self):
        """Returns the Block LDU decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (L, D, U) : Matrices
            L : Lower Diagonal Matrix
            D : Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, D, U = X.LDUdecomposition()
        >>> block_collapse(L*D*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            try:
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LDU decomposition cannot be calculated when                    "A" is singular')
            Ip = Identity(B.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            L = BlockMatrix([[Ip, Z], [C * AI, Iq]])
            D = BlockDiagMatrix(A, self.schur())
            U = BlockMatrix([[Ip, AI * B], [Z.T, Iq]])
            return (L, D, U)
        else:
            raise ShapeError('Block LDU decomposition is supported only for 2x2 block matrices')

    def UDLdecomposition(self):
        """Returns the Block UDL decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (U, D, L) : Matrices
            U : Upper Diagonal Matrix
            D : Diagonal Matrix
            L : Lower Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> U, D, L = X.UDLdecomposition()
        >>> block_collapse(U*D*L)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "D" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            try:
                DI = D.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block UDL decomposition cannot be calculated when                    "D" is singular')
            Ip = Identity(A.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            U = BlockMatrix([[Ip, B * DI], [Z.T, Iq]])
            D = BlockDiagMatrix(self.schur('D'), D)
            L = BlockMatrix([[Ip, Z], [DI * C, Iq]])
            return (U, D, L)
        else:
            raise ShapeError('Block UDL decomposition is supported only for 2x2 block matrices')

    def LUdecomposition(self):
        """Returns the Block LU decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (L, U) : Matrices
            L : Lower Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, U = X.LUdecomposition()
        >>> block_collapse(L*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        """
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            try:
                A = A ** 0.5
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LU decomposition cannot be calculated when                    "A" is singular')
            Z = ZeroMatrix(*B.shape)
            Q = self.schur() ** 0.5
            L = BlockMatrix([[A, Z], [C * AI, Q]])
            U = BlockMatrix([[A, AI * B], [Z.T, Q]])
            return (L, U)
        else:
            raise ShapeError('Block LU decomposition is supported only for 2x2 block matrices')

    def _entry(self, i, j, **kwargs):
        orig_i, orig_j = (i, j)
        for row_block, numrows in enumerate(self.rowblocksizes):
            cmp = i < numrows
            if cmp == True:
                break
            elif cmp == False:
                i -= numrows
            elif row_block < self.blockshape[0] - 1:
                return MatrixElement(self, orig_i, orig_j)
        for col_block, numcols in enumerate(self.colblocksizes):
            cmp = j < numcols
            if cmp == True:
                break
            elif cmp == False:
                j -= numcols
            elif col_block < self.blockshape[1] - 1:
                return MatrixElement(self, orig_i, orig_j)
        return self.blocks[row_block, col_block][i, j]

    @property
    def is_Identity(self):
        if self.blockshape[0] != self.blockshape[1]:
            return False
        for i in range(self.blockshape[0]):
            for j in range(self.blockshape[1]):
                if i == j and (not self.blocks[i, j].is_Identity):
                    return False
                if i != j and (not self.blocks[i, j].is_ZeroMatrix):
                    return False
        return True

    @property
    def is_structurally_symmetric(self):
        return self.rowblocksizes == self.colblocksizes

    def equals(self, other):
        if self == other:
            return True
        if isinstance(other, BlockMatrix) and self.blocks == other.blocks:
            return True
        return super().equals(other)