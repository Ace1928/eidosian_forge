import numpy as np
from scipy._lib._util import _asarray_validated
from ._misc import norm
from .lapack import ztrsyl, dtrsyl
from ._decomp_schur import schur, rsf2csf
from ._matfuncs_sqrtm_triu import within_block_loop  # noqa: E402
def _sqrtm_triu(T, blocksize=64):
    """
    Matrix square root of an upper triangular matrix.

    This is a helper function for `sqrtm` and `logm`.

    Parameters
    ----------
    T : (N, N) array_like upper triangular
        Matrix whose square root to evaluate
    blocksize : int, optional
        If the blocksize is not degenerate with respect to the
        size of the input array, then use a blocked algorithm. (Default: 64)

    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `T`

    References
    ----------
    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
           "Blocked Schur Algorithms for Computing the Matrix Square Root,
           Lecture Notes in Computer Science, 7782. pp. 171-182.

    """
    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0
    if not keep_it_real:
        T = np.asarray(T, dtype=np.complex128, order='C')
        T_diag = np.asarray(T_diag, dtype=np.complex128)
    else:
        T = np.asarray(T, dtype=np.float64, order='C')
        T_diag = np.asarray(T_diag, dtype=np.float64)
    R = np.diag(np.sqrt(T_diag))
    n, n = T.shape
    nblocks = max(n // blocksize, 1)
    bsmall, nlarge = divmod(n, nblocks)
    blarge = bsmall + 1
    nsmall = nblocks - nlarge
    if nsmall * bsmall + nlarge * blarge != n:
        raise Exception('internal inconsistency')
    start_stop_pairs = []
    start = 0
    for count, size in ((nsmall, bsmall), (nlarge, blarge)):
        for i in range(count):
            start_stop_pairs.append((start, start + size))
            start += size
    try:
        within_block_loop(R, T, start_stop_pairs, nblocks)
    except RuntimeError as e:
        raise SqrtmError(*e.args) from e
    for j in range(nblocks):
        jstart, jstop = start_stop_pairs[j]
        for i in range(j - 1, -1, -1):
            istart, istop = start_stop_pairs[i]
            S = T[istart:istop, jstart:jstop]
            if j - i > 1:
                S = S - R[istart:istop, istop:jstart].dot(R[istop:jstart, jstart:jstop])
            Rii = R[istart:istop, istart:istop]
            Rjj = R[jstart:jstop, jstart:jstop]
            if keep_it_real:
                x, scale, info = dtrsyl(Rii, Rjj, S)
            else:
                x, scale, info = ztrsyl(Rii, Rjj, S)
            R[istart:istop, jstart:jstop] = x * scale
    return R