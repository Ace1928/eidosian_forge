import numpy as np
from scipy.linalg import solve
def _nnls(A, b, maxiter=None, tol=None):
    """
    This is a single RHS algorithm from ref [2] above. For multiple RHS
    support, the algorithm is given in  :doi:`10.1002/cem.889`
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = b @ A
    if not maxiter:
        maxiter = 3 * n
    if tol is None:
        tol = 10 * max(m, n) * np.spacing(1.0)
    x = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=bool)
    resid = Atb.copy().astype(np.float64)
    iter = 0
    while not P.all() and (resid[~P] > tol).any():
        resid[P] = -np.inf
        k = np.argmax(resid)
        P[k] = True
        s = np.zeros(n, dtype=np.float64)
        P_ind = P.nonzero()[0]
        s[P] = solve(AtA[P_ind[:, None], P_ind[None, :]], Atb[P], assume_a='sym', check_finite=False)
        while iter < maxiter and s[P].min() <= tol:
            alpha_ind = ((s < tol) & P).nonzero()
            alpha = (x[alpha_ind] / (x[alpha_ind] - s[alpha_ind])).min()
            x *= 1 - alpha
            x += alpha * s
            P[x < tol] = False
            s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym', check_finite=False)
            s[~P] = 0
            iter += 1
        x[:] = s[:]
        resid = Atb - AtA @ x
        if iter == maxiter:
            return (x, 0.0, -1)
    return (x, np.linalg.norm(A @ x - b), 1)