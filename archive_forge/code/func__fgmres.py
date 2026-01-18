import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import (get_blas_funcs, qr, solve, svd, qr_insert, lstsq)
from .iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve.utils import make_system
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def _fgmres(matvec, v0, m, atol, lpsolve=None, rpsolve=None, cs=(), outer_v=(), prepend_outer_v=False):
    """
    FGMRES Arnoldi process, with optional projection or augmentation

    Parameters
    ----------
    matvec : callable
        Operation A*x
    v0 : ndarray
        Initial vector, normalized to nrm2(v0) == 1
    m : int
        Number of GMRES rounds
    atol : float
        Absolute tolerance for early exit
    lpsolve : callable
        Left preconditioner L
    rpsolve : callable
        Right preconditioner R
    cs : list of (ndarray, ndarray)
        Columns of matrices C and U in GCROT
    outer_v : list of ndarrays
        Augmentation vectors in LGMRES
    prepend_outer_v : bool, optional
        Whether augmentation vectors come before or after
        Krylov iterates

    Raises
    ------
    LinAlgError
        If nans encountered

    Returns
    -------
    Q, R : ndarray
        QR decomposition of the upper Hessenberg H=QR
    B : ndarray
        Projections corresponding to matrix C
    vs : list of ndarray
        Columns of matrix V
    zs : list of ndarray
        Columns of matrix Z
    y : ndarray
        Solution to ||H y - e_1||_2 = min!
    res : float
        The final (preconditioned) residual norm

    """
    if lpsolve is None:

        def lpsolve(x):
            return x
    if rpsolve is None:

        def rpsolve(x):
            return x
    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (v0,))
    vs = [v0]
    zs = []
    y = None
    res = np.nan
    m = m + len(outer_v)
    B = np.zeros((len(cs), m), dtype=v0.dtype)
    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)
    eps = np.finfo(v0.dtype).eps
    breakdown = False
    for j in range(m):
        if prepend_outer_v and j < len(outer_v):
            z, w = outer_v[j]
        elif prepend_outer_v and j == len(outer_v):
            z = rpsolve(v0)
            w = None
        elif not prepend_outer_v and j >= m - len(outer_v):
            z, w = outer_v[j - (m - len(outer_v))]
        else:
            z = rpsolve(vs[-1])
            w = None
        if w is None:
            w = lpsolve(matvec(z))
        else:
            w = w.copy()
        w_norm = nrm2(w)
        for i, c in enumerate(cs):
            alpha = dot(c, w)
            B[i, j] = alpha
            w = axpy(c, w, c.shape[0], -alpha)
        hcur = np.zeros(j + 2, dtype=Q.dtype)
        for i, v in enumerate(vs):
            alpha = dot(v, w)
            hcur[i] = alpha
            w = axpy(v, w, v.shape[0], -alpha)
        hcur[i + 1] = nrm2(w)
        with np.errstate(over='ignore', divide='ignore'):
            alpha = 1 / hcur[-1]
        if np.isfinite(alpha):
            w = scal(alpha, w)
        if not hcur[-1] > eps * w_norm:
            breakdown = True
        vs.append(w)
        zs.append(z)
        Q2 = np.zeros((j + 2, j + 2), dtype=Q.dtype, order='F')
        Q2[:j + 1, :j + 1] = Q
        Q2[j + 1, j + 1] = 1
        R2 = np.zeros((j + 2, j), dtype=R.dtype, order='F')
        R2[:j + 1, :] = R
        Q, R = qr_insert(Q2, R2, hcur, j, which='col', overwrite_qru=True, check_finite=False)
        res = abs(Q[0, -1])
        if res < atol or breakdown:
            break
    if not np.isfinite(R[j, j]):
        raise LinAlgError()
    y, _, _, _ = lstsq(R[:j + 1, :j + 1], Q[0, :j + 1].conj())
    B = B[:, :j + 1]
    return (Q, R, B, vs, zs, y, res)