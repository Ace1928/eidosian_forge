import random
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .common import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve
class DenseMatrix(RepMatrix):
    """Matrix implementation based on DomainMatrix as the internal representation"""
    is_MatrixExpr = False
    _op_priority = 10.01
    _class_priority = 4

    @property
    def _mat(self):
        sympy_deprecation_warning('\n            The private _mat attribute of Matrix is deprecated. Use the\n            .flat() method instead.\n            ', deprecated_since_version='1.9', active_deprecations_target='deprecated-private-matrix-attributes')
        return self.flat()

    def _eval_inverse(self, **kwargs):
        return self.inv(method=kwargs.get('method', 'GE'), iszerofunc=kwargs.get('iszerofunc', _iszero), try_block_diag=kwargs.get('try_block_diag', False))

    def as_immutable(self):
        """Returns an Immutable version of this Matrix
        """
        from .immutable import ImmutableDenseMatrix as cls
        return cls._fromrep(self._rep.copy())

    def as_mutable(self):
        """Returns a mutable version of this matrix

        Examples
        ========

        >>> from sympy import ImmutableMatrix
        >>> X = ImmutableMatrix([[1, 2], [3, 4]])
        >>> Y = X.as_mutable()
        >>> Y[1, 1] = 5 # Can set values in Y
        >>> Y
        Matrix([
        [1, 2],
        [3, 5]])
        """
        return Matrix(self)

    def cholesky(self, hermitian=True):
        return _cholesky(self, hermitian=hermitian)

    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition(self, hermitian=hermitian)

    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve(self, rhs)

    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve(self, rhs)
    cholesky.__doc__ = _cholesky.__doc__
    LDLdecomposition.__doc__ = _LDLdecomposition.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__