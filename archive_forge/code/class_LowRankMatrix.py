import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
class LowRankMatrix:
    """
    A matrix represented as

    .. math:: \\alpha I + \\sum_{n=0}^{n=M} c_n d_n^\\dagger

    However, if the rank of the matrix reaches the dimension of the vectors,
    full matrix representation will be used thereon.

    """

    def __init__(self, alpha, n, dtype):
        self.alpha = alpha
        self.cs = []
        self.ds = []
        self.n = n
        self.dtype = dtype
        self.collapsed = None

    @staticmethod
    def _matvec(v, alpha, cs, ds):
        axpy, scal, dotc = get_blas_funcs(['axpy', 'scal', 'dotc'], cs[:1] + [v])
        w = alpha * v
        for c, d in zip(cs, ds):
            a = dotc(d, v)
            w = axpy(c, w, w.size, a)
        return w

    @staticmethod
    def _solve(v, alpha, cs, ds):
        """Evaluate w = M^-1 v"""
        if len(cs) == 0:
            return v / alpha
        axpy, dotc = get_blas_funcs(['axpy', 'dotc'], cs[:1] + [v])
        c0 = cs[0]
        A = alpha * np.identity(len(cs), dtype=c0.dtype)
        for i, d in enumerate(ds):
            for j, c in enumerate(cs):
                A[i, j] += dotc(d, c)
        q = np.zeros(len(cs), dtype=c0.dtype)
        for j, d in enumerate(ds):
            q[j] = dotc(d, v)
        q /= alpha
        q = solve(A, q)
        w = v / alpha
        for c, qc in zip(cs, q):
            w = axpy(c, w, w.size, -qc)
        return w

    def matvec(self, v):
        """Evaluate w = M v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed, v)
        return LowRankMatrix._matvec(v, self.alpha, self.cs, self.ds)

    def rmatvec(self, v):
        """Evaluate w = M^H v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed.T.conj(), v)
        return LowRankMatrix._matvec(v, np.conj(self.alpha), self.ds, self.cs)

    def solve(self, v, tol=0):
        """Evaluate w = M^-1 v"""
        if self.collapsed is not None:
            return solve(self.collapsed, v)
        return LowRankMatrix._solve(v, self.alpha, self.cs, self.ds)

    def rsolve(self, v, tol=0):
        """Evaluate w = M^-H v"""
        if self.collapsed is not None:
            return solve(self.collapsed.T.conj(), v)
        return LowRankMatrix._solve(v, np.conj(self.alpha), self.ds, self.cs)

    def append(self, c, d):
        if self.collapsed is not None:
            self.collapsed += c[:, None] * d[None, :].conj()
            return
        self.cs.append(c)
        self.ds.append(d)
        if len(self.cs) > c.size:
            self.collapse()

    def __array__(self):
        if self.collapsed is not None:
            return self.collapsed
        Gm = self.alpha * np.identity(self.n, dtype=self.dtype)
        for c, d in zip(self.cs, self.ds):
            Gm += c[:, None] * d[None, :].conj()
        return Gm

    def collapse(self):
        """Collapse the low-rank matrix to a full-rank one."""
        self.collapsed = np.array(self)
        self.cs = None
        self.ds = None
        self.alpha = None

    def restart_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping all vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        if len(self.cs) > rank:
            del self.cs[:]
            del self.ds[:]

    def simple_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping oldest vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        while len(self.cs) > rank:
            del self.cs[0]
            del self.ds[0]

    def svd_reduce(self, max_rank, to_retain=None):
        """
        Reduce the rank of the matrix by retaining some SVD components.

        This corresponds to the "Broyden Rank Reduction Inverse"
        algorithm described in [1]_.

        Note that the SVD decomposition can be done by solving only a
        problem whose size is the effective rank of this matrix, which
        is viable even for large problems.

        Parameters
        ----------
        max_rank : int
            Maximum rank of this matrix after reduction.
        to_retain : int, optional
            Number of SVD components to retain when reduction is done
            (ie. rank > max_rank). Default is ``max_rank - 2``.

        References
        ----------
        .. [1] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional
           systems of nonlinear equations". Mathematisch Instituut,
           Universiteit Leiden, The Netherlands (2003).

           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

        """
        if self.collapsed is not None:
            return
        p = max_rank
        if to_retain is not None:
            q = to_retain
        else:
            q = p - 2
        if self.cs:
            p = min(p, len(self.cs[0]))
        q = max(0, min(q, p - 1))
        m = len(self.cs)
        if m < p:
            return
        C = np.array(self.cs).T
        D = np.array(self.ds).T
        D, R = qr(D, mode='economic')
        C = dot(C, R.T.conj())
        U, S, WH = svd(C, full_matrices=False)
        C = dot(C, inv(WH))
        D = dot(D, WH.T.conj())
        for k in range(q):
            self.cs[k] = C[:, k].copy()
            self.ds[k] = D[:, k].copy()
        del self.cs[q:]
        del self.ds[q:]