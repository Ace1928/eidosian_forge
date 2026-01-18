import numpy as np
from scipy.sparse.linalg import aslinearoperator
def _onenormest_core(A, AT, t, itmax):
    """
    Compute a lower bound of the 1-norm of a sparse matrix.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
    itmax : int, optional
        Use at most this many iterations.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.
    nmults : int, optional
        The number of matrix products that were computed.
    nresamples : int, optional
        The number of times a parallel column was observed,
        necessitating a re-randomization of the column.

    Notes
    -----
    This is algorithm 2.4.

    """
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    if itmax < 2:
        raise ValueError('at least two iterations are required')
    if t < 1:
        raise ValueError('at least one column is required')
    n = A.shape[0]
    if t >= n:
        raise ValueError('t should be smaller than the order of A')
    nmults = 0
    nresamples = 0
    X = np.ones((n, t), dtype=float)
    if t > 1:
        for i in range(1, t):
            resample_column(i, X)
        for i in range(t):
            while column_needs_resampling(i, X):
                resample_column(i, X)
                nresamples += 1
    X /= float(n)
    ind_hist = np.zeros(0, dtype=np.intp)
    est_old = 0
    S = np.zeros((n, t), dtype=float)
    k = 1
    ind = None
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        nmults += 1
        mags = _sum_abs_axis0(Y)
        est = np.max(mags)
        best_j = np.argmax(mags)
        if est > est_old or k == 2:
            if k >= 2:
                ind_best = ind[best_j]
            w = Y[:, best_j]
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        S_old = S
        if k > itmax:
            break
        S = sign_round_up(Y)
        del Y
        if every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old):
            break
        if t > 1:
            for i in range(t):
                while column_needs_resampling(i, S, S_old):
                    resample_column(i, S)
                    nresamples += 1
        del S_old
        Z = np.asarray(AT_linear_operator.matmat(S))
        nmults += 1
        h = _max_abs_axis1(Z)
        del Z
        if k >= 2 and max(h) == h[ind_best]:
            break
        ind = np.argsort(h)[::-1][:t + len(ind_hist)].copy()
        del h
        if t > 1:
            if np.isin(ind[:t], ind_hist).all():
                break
            seen = np.isin(ind, ind_hist)
            ind = np.concatenate((ind[~seen], ind[seen]))
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])
        new_ind = ind[:t][~np.isin(ind[:t], ind_hist)]
        ind_hist = np.concatenate((ind_hist, new_ind))
        k += 1
    v = elementary_vector(n, ind_best)
    return (est, v, w, nmults, nresamples)