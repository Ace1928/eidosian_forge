from __future__ import annotations
from functools import wraps
from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.matrices import MatrixKind, MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant
class MatrixExpr(Expr):
    """Superclass for Matrix Expressions

    MatrixExprs represent abstract matrices, linear transformations represented
    within a particular basis.

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========

    MatrixSymbol, MatAdd, MatMul, Transpose, Inverse
    """
    __slots__: tuple[str, ...] = ()
    _iterable = False
    _op_priority = 11.0
    is_Matrix: bool = True
    is_MatrixExpr: bool = True
    is_Identity: FuzzyBool = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False
    is_commutative = False
    is_number = False
    is_symbol = False
    is_scalar = False
    kind: MatrixKind = MatrixKind()

    def __new__(cls, *args, **kwargs):
        args = map(_sympify, args)
        return Basic.__new__(cls, *args, **kwargs)

    @property
    def shape(self) -> tuple[Expr, Expr]:
        raise NotImplementedError

    @property
    def _add_handler(self):
        return MatAdd

    @property
    def _mul_handler(self):
        return MatMul

    def __neg__(self):
        return MatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return MatAdd(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return MatAdd(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return MatAdd(other, -self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return MatPow(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        raise NotImplementedError('Matrix Power not defined')

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * other ** S.NegativeOne

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        raise NotImplementedError()

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    @property
    def is_square(self) -> bool | None:
        rows, cols = self.shape
        if isinstance(rows, Integer) and isinstance(cols, Integer):
            return rows == cols
        if rows == cols:
            return True
        return None

    def _eval_conjugate(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(Transpose(self))

    def as_real_imag(self, deep=True, **hints):
        return self._eval_as_real_imag()

    def _eval_as_real_imag(self):
        real = S.Half * (self + self._eval_conjugate())
        im = (self - self._eval_conjugate()) / (2 * S.ImaginaryUnit)
        return (real, im)

    def _eval_inverse(self):
        return Inverse(self)

    def _eval_determinant(self):
        return Determinant(self)

    def _eval_transpose(self):
        return Transpose(self)

    def _eval_power(self, exp):
        """
        Override this in sub-classes to implement simplification of powers.  The cases where the exponent
        is -1, 0, 1 are already covered in MatPow.doit(), so implementations can exclude these cases.
        """
        return MatPow(self, exp)

    def _eval_simplify(self, **kwargs):
        if self.is_Atom:
            return self
        else:
            from sympy.simplify import simplify
            return self.func(*[simplify(x, **kwargs) for x in self.args])

    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)

    def _eval_derivative_n_times(self, x, n):
        return Basic._eval_derivative_n_times(self, x, n)

    def _eval_derivative(self, x):
        if self.has(x):
            return super()._eval_derivative(x)
        else:
            return ZeroMatrix(*self.shape)

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = check_assumptions(dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError('The dimension specification {} should be a nonnegative integer.'.format(dim))

    def _entry(self, i, j, **kwargs):
        raise NotImplementedError('Indexing not implemented for %s' % self.__class__.__name__)

    def adjoint(self):
        return adjoint(self)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return (S.One, self)

    def conjugate(self):
        return conjugate(self)

    def transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return transpose(self)

    @property
    def T(self):
        """Matrix transposition"""
        return self.transpose()

    def inverse(self):
        if self.is_square is False:
            raise NonSquareMatrixError('Inverse of non-square matrix')
        return self._eval_inverse()

    def inv(self):
        return self.inverse()

    def det(self):
        from sympy.matrices.expressions.determinant import det
        return det(self)

    @property
    def I(self):
        return self.inverse()

    def valid_index(self, i, j):

        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return is_valid(i) and is_valid(j) and (self.rows is None or ((i >= -self.rows) != False and (i < self.rows) != False)) and ((j >= -self.cols) != False) and ((j < self.cols) != False)

    def __getitem__(self, key):
        if not isinstance(key, tuple) and isinstance(key, slice):
            from sympy.matrices.expressions.slice import MatrixSlice
            return MatrixSlice(self, key, (0, None, 1))
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                from sympy.matrices.expressions.slice import MatrixSlice
                return MatrixSlice(self, i, j)
            i, j = (_sympify(i), _sympify(j))
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError('Invalid indices (%s, %s)' % (i, j))
        elif isinstance(key, (SYMPY_INTS, Integer)):
            rows, cols = self.shape
            if not isinstance(cols, Integer):
                raise IndexError(filldedent('\n                    Single indexing is only supported when the number\n                    of columns is known.'))
            key = _sympify(key)
            i = key // cols
            j = key % cols
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError('Invalid index %s' % key)
        elif isinstance(key, (Symbol, Expr)):
            raise IndexError(filldedent('\n                Only integers may be used when addressing the matrix\n                with a single index.'))
        raise IndexError('Invalid index, wanted %s[i,j]' % self)

    def _is_shape_symbolic(self) -> bool:
        return not isinstance(self.rows, (SYMPY_INTS, Integer)) or not isinstance(self.cols, (SYMPY_INTS, Integer))

    def as_explicit(self):
        """
        Returns a dense Matrix with elements represented explicitly

        Returns an object of type ImmutableDenseMatrix.

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.as_explicit()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_mutable: returns mutable Matrix type

        """
        if self._is_shape_symbolic():
            raise ValueError('Matrix with symbolic shape cannot be represented explicitly.')
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix([[self[i, j] for j in range(self.cols)] for i in range(self.rows)])

    def as_mutable(self):
        """
        Returns a dense, mutable matrix with elements represented explicitly

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.shape
        (3, 3)
        >>> I.as_mutable()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_explicit: returns ImmutableDenseMatrix
        """
        return self.as_explicit().as_mutable()

    def __array__(self):
        from numpy import empty
        a = empty(self.shape, dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                a[i, j] = self[i, j]
        return a

    def equals(self, other):
        """
        Test elementwise equality between matrices, potentially of different
        types

        >>> from sympy import Identity, eye
        >>> Identity(3).equals(eye(3))
        True
        """
        return self.as_explicit().equals(other)

    def canonicalize(self):
        return self

    def as_coeff_mmul(self):
        return (S.One, MatMul(self))

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None, dimensions=None):
        """
        Parse expression of matrices with explicitly summed indices into a
        matrix expression without indices, if possible.

        This transformation expressed in mathematical notation:

        `\\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \\Longrightarrow \\mathbf{A}\\cdot \\mathbf{B}`

        Optional parameter ``first_index``: specify which free index to use as
        the index starting the expression.

        Examples
        ========

        >>> from sympy import MatrixSymbol, MatrixExpr, Sum
        >>> from sympy.abc import i, j, k, l, N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B

        Transposition is detected:

        >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A.T*B

        Detect the trace:

        >>> expr = Sum(A[i, i], (i, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        Trace(A)

        More complicated expressions:

        >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B.T*A.T
        """
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
        first_indices = []
        if first_index is not None:
            first_indices.append(first_index)
        if last_index is not None:
            first_indices.append(last_index)
        arr = convert_indexed_to_array(expr, first_indices=first_indices)
        return convert_array_to_matrix(arr)

    def applyfunc(self, func):
        from .applyfunc import ElementwiseApplyFunction
        return ElementwiseApplyFunction(func, self)