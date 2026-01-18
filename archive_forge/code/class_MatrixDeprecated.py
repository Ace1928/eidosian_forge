import mpmath as mp
from collections.abc import Callable
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import diff
from sympy.core.expr import Expr
from sympy.core.kind import _NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify, _sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import flatten, NotIterable, is_sequence, reshape
from sympy.utilities.misc import as_int, filldedent
from .common import (
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify
from .determinant import (
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import (
from .decompositions import (
from .graph import (
from .solvers import (
from .inverse import (
class MatrixDeprecated(MatrixCommon):
    """A class to house deprecated matrix methods."""

    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_simplify):
        return self.charpoly(x=x)

    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        berkowitz
        """
        return self.det(method='berkowitz')

    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        return self.eigenvals(**flags)

    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        sign, minors = (self.one, [])
        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign
        return tuple(minors)

    def berkowitz(self):
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk
        if not self.is_square:
            raise NonSquareMatrixError()
        A, N = (self, self.rows)
        transforms = [0] * (N - 1)
        for n in range(N, 1, -1):
            T, k = (zeros(n + 1, n), n - 1)
            R, C = (-A[k, :k], A[:k, k])
            A, a = (A[:k, :k], -A[k, k])
            items = [C]
            for i in range(0, n - 2):
                items.append(A * items[i])
            for i, B in enumerate(items):
                items[i] = (R * B)[0, 0]
            items = [self.one, a] + items
            for i in range(n):
                T[i:, i] = items[:n - i + 1]
            transforms[k - 1] = T
        polys = [self._new([self.one, -A[0, 0]])]
        for i, T in enumerate(transforms):
            polys.append(T * polys[i])
        return berk + tuple(map(tuple, polys))

    def cofactorMatrix(self, method='berkowitz'):
        return self.cofactor_matrix(method=method)

    def det_bareis(self):
        return _det_bareiss(self)

    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition.


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.

        See Also
        ========


        det
        det_bareiss
        berkowitz_det
        """
        return self.det(method='lu')

    def jordan_cell(self, eigenval, n):
        return self.jordan_block(size=n, eigenvalue=eigenval)

    def jordan_cells(self, calc_transformation=True):
        P, J = self.jordan_form()
        return (P, J.get_diag_blocks())

    def minorEntry(self, i, j, method='berkowitz'):
        return self.minor(i, j, method=method)

    def minorMatrix(self, i, j):
        return self.minor_submatrix(i, j)

    def permuteBkwd(self, perm):
        """Permute the rows of the matrix with the given permutation in reverse."""
        return self.permute_rows(perm, direction='backward')

    def permuteFwd(self, perm):
        """Permute the rows of the matrix with the given permutation."""
        return self.permute_rows(perm, direction='forward')