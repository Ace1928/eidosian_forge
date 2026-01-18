from functools import reduce
from math import prod
from sympy.core import Mul, sympify
from sympy.functions import adjoint
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.expressions.special import Identity
from sympy.matrices.matrices import MatrixBase
from sympy.strategies import (
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift
from .matadd import MatAdd
from .matmul import MatMul
from .matpow import MatPow
class KroneckerProduct(MatrixExpr):
    """
    The Kronecker product of two or more arguments.

    The Kronecker product is a non-commutative product of matrices.
    Given two matrices of dimension (m, n) and (s, t) it produces a matrix
    of dimension (m s, n t).

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``kronecker_product()`` or call the ``.doit()`` or  ``.as_explicit()``
    methods.

    >>> from sympy import KroneckerProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(KroneckerProduct(A, B), KroneckerProduct)
    True
    """
    is_KroneckerProduct = True

    def __new__(cls, *args, check=True):
        args = list(map(sympify, args))
        if all((a.is_Identity for a in args)):
            ret = Identity(prod((a.rows for a in args)))
            if all((isinstance(a, MatrixBase) for a in args)):
                return ret.as_explicit()
            else:
                return ret
        if check:
            validate(*args)
        return super().__new__(cls, *args)

    @property
    def shape(self):
        rows, cols = self.args[0].shape
        for mat in self.args[1:]:
            rows *= mat.rows
            cols *= mat.cols
        return (rows, cols)

    def _entry(self, i, j, **kwargs):
        result = 1
        for mat in reversed(self.args):
            i, m = divmod(i, mat.rows)
            j, n = divmod(j, mat.cols)
            result *= mat[m, n]
        return result

    def _eval_adjoint(self):
        return KroneckerProduct(*list(map(adjoint, self.args))).doit()

    def _eval_conjugate(self):
        return KroneckerProduct(*[a.conjugate() for a in self.args]).doit()

    def _eval_transpose(self):
        return KroneckerProduct(*list(map(transpose, self.args))).doit()

    def _eval_trace(self):
        from .trace import trace
        return Mul(*[trace(a) for a in self.args])

    def _eval_determinant(self):
        from .determinant import det, Determinant
        if not all((a.is_square for a in self.args)):
            return Determinant(self)
        m = self.rows
        return Mul(*[det(a) ** (m / a.rows) for a in self.args])

    def _eval_inverse(self):
        try:
            return KroneckerProduct(*[a.inverse() for a in self.args])
        except ShapeError:
            from sympy.matrices.expressions.inverse import Inverse
            return Inverse(self)

    def structurally_equal(self, other):
        """Determine whether two matrices have the same Kronecker product structure

        Examples
        ========

        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols
        >>> m, n = symbols(r'm, n', integer=True)
        >>> A = MatrixSymbol('A', m, m)
        >>> B = MatrixSymbol('B', n, n)
        >>> C = MatrixSymbol('C', m, m)
        >>> D = MatrixSymbol('D', n, n)
        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(C, D))
        True
        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(D, C))
        False
        >>> KroneckerProduct(A, B).structurally_equal(C)
        False
        """
        return isinstance(other, KroneckerProduct) and self.shape == other.shape and (len(self.args) == len(other.args)) and all((a.shape == b.shape for a, b in zip(self.args, other.args)))

    def has_matching_shape(self, other):
        """Determine whether two matrices have the appropriate structure to bring matrix
        multiplication inside the KroneckerProdut

        Examples
        ========
        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols
        >>> m, n = symbols(r'm, n', integer=True)
        >>> A = MatrixSymbol('A', m, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(B, A))
        True
        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(A, B))
        False
        >>> KroneckerProduct(A, B).has_matching_shape(A)
        False
        """
        return isinstance(other, KroneckerProduct) and self.cols == other.rows and (len(self.args) == len(other.args)) and all((a.cols == b.rows for a, b in zip(self.args, other.args)))

    def _eval_expand_kroneckerproduct(self, **hints):
        return flatten(canon(typed({KroneckerProduct: distribute(KroneckerProduct, MatAdd)}))(self))

    def _kronecker_add(self, other):
        if self.structurally_equal(other):
            return self.__class__(*[a + b for a, b in zip(self.args, other.args)])
        else:
            return self + other

    def _kronecker_mul(self, other):
        if self.has_matching_shape(other):
            return self.__class__(*[a * b for a, b in zip(self.args, other.args)])
        else:
            return self * other

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return canonicalize(KroneckerProduct(*args))