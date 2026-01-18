import copy
from sympy.core import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.miscellaneous import Min, sqrt
from sympy.functions.elementary.complexes import sign
from .common import NonSquareMatrixError, NonPositiveDefiniteMatrixError
from .utilities import _get_intermediate_simp, _iszero
from .determinant import _find_reasonable_pivot_naive
def _QRdecomposition_optional(M, normalize=True):

    def dot(u, v):
        return u.dot(v, hermitian=True)
    dps = _get_intermediate_simp(expand_mul, expand_mul)
    A = M.as_mutable()
    ranked = []
    Q = A
    R = A.zeros(A.cols)
    for j in range(A.cols):
        for i in range(j):
            if Q[:, i].is_zero_matrix:
                continue
            R[i, j] = dot(Q[:, i], Q[:, j]) / dot(Q[:, i], Q[:, i])
            R[i, j] = dps(R[i, j])
            Q[:, j] -= Q[:, i] * R[i, j]
        Q[:, j] = dps(Q[:, j])
        if Q[:, j].is_zero_matrix is not True:
            ranked.append(j)
            R[j, j] = M.one
    Q = Q.extract(range(Q.rows), ranked)
    R = R.extract(ranked, range(R.cols))
    if normalize:
        for i in range(Q.cols):
            norm = Q[:, i].norm()
            Q[:, i] /= norm
            R[i, :] *= norm
    return (M.__class__(Q), M.__class__(R))