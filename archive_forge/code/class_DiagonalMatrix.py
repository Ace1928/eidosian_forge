from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core import S, Eq, Ge
from sympy.core.mul import Mul
from sympy.functions.special.tensor_functions import KroneckerDelta
class DiagonalMatrix(MatrixExpr):
    """DiagonalMatrix(M) will create a matrix expression that
    behaves as though all off-diagonal elements,
    `M[i, j]` where `i != j`, are zero.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalMatrix, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> D = DiagonalMatrix(MatrixSymbol('x', 2, 3))
    >>> D[1, 2]
    0
    >>> D[1, 1]
    x[1, 1]

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> D.diagonal_length
    2
    >>> DiagonalMatrix(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When one of the dimensions is symbolic the other will be treated as
    though it is smaller:

    >>> tall = DiagonalMatrix(MatrixSymbol('x', n, 3))
    >>> tall.diagonal_length
    3
    >>> tall[10, 1]
    0

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalMatrix(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    arg = property(lambda self: self.args[0])
    shape = property(lambda self: self.arg.shape)

    @property
    def diagonal_length(self):
        r, c = self.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and (not c.is_Integer):
            m = r
        elif c.is_Integer and (not r.is_Integer):
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m

    def _entry(self, i, j, **kwargs):
        if self.diagonal_length is not None:
            if Ge(i, self.diagonal_length) is S.true:
                return S.Zero
            elif Ge(j, self.diagonal_length) is S.true:
                return S.Zero
        eq = Eq(i, j)
        if eq is S.true:
            return self.arg[i, i]
        elif eq is S.false:
            return S.Zero
        return self.arg[i, j] * KroneckerDelta(i, j)