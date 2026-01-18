from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def _row_reduce(M, iszerofunc, simpfunc, normalize_last=True, normalize=True, zero_above=True):
    mat, pivot_cols, swaps = _row_reduce_list(list(M), M.rows, M.cols, M.one, iszerofunc, simpfunc, normalize_last=normalize_last, normalize=normalize, zero_above=zero_above)
    return (M._new(M.rows, M.cols, mat), pivot_cols, swaps)