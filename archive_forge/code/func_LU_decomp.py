from copy import copy
from ..libmp.backend import xrange
def LU_decomp(ctx, A, overwrite=False, use_cache=True):
    """
        LU-factorization of a n*n matrix using the Gauss algorithm.
        Returns L and U in one matrix and the pivot indices.

        Use overwrite to specify whether A will be overwritten with L and U.
        """
    if not A.rows == A.cols:
        raise ValueError('need n*n matrix')
    if use_cache and isinstance(A, ctx.matrix) and A._LU:
        return A._LU
    if not overwrite:
        orig = A
        A = A.copy()
    tol = ctx.absmin(ctx.mnorm(A, 1) * ctx.eps)
    n = A.rows
    p = [None] * (n - 1)
    for j in xrange(n - 1):
        biggest = 0
        for k in xrange(j, n):
            s = ctx.fsum([ctx.absmin(A[k, l]) for l in xrange(j, n)])
            if ctx.absmin(s) <= tol:
                raise ZeroDivisionError('matrix is numerically singular')
            current = 1 / s * ctx.absmin(A[k, j])
            if current > biggest:
                biggest = current
                p[j] = k
        ctx.swap_row(A, j, p[j])
        if ctx.absmin(A[j, j]) <= tol:
            raise ZeroDivisionError('matrix is numerically singular')
        for i in xrange(j + 1, n):
            A[i, j] /= A[j, j]
            for k in xrange(j + 1, n):
                A[i, k] -= A[i, j] * A[j, k]
    if ctx.absmin(A[n - 1, n - 1]) <= tol:
        raise ZeroDivisionError('matrix is numerically singular')
    if not overwrite and isinstance(orig, ctx.matrix):
        orig._LU = (A, p)
    return (A, p)