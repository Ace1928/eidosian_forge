from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def _permute_complexity_right(M, iszerofunc):
    """Permute columns with complicated elements as
        far right as they can go.  Since the ``sympy`` row reduction
        algorithms start on the left, having complexity right-shifted
        speeds things up.

        Returns a tuple (mat, perm) where perm is a permutation
        of the columns to perform to shift the complex columns right, and mat
        is the permuted matrix."""

    def complexity(i):
        return sum((1 if iszerofunc(e) is None else 0 for e in M[:, i]))
    complex = [(complexity(i), i) for i in range(M.cols)]
    perm = [j for i, j in sorted(complex)]
    return (M.permute(perm, orientation='cols'), perm)