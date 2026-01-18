import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
class _MWU:
    """Distribution of MWU statistic under the null hypothesis"""

    def __init__(self):
        """Minimal initializer"""
        self._fmnks = -np.ones((1, 1, 1))
        self._recursive = None

    def pmf(self, k, m, n):
        if self._recursive is None and m <= 500 and (n <= 500) or self._recursive:
            return self.pmf_recursive(k, m, n)
        else:
            return self.pmf_iterative(k, m, n)

    def pmf_recursive(self, k, m, n):
        """Probability mass function, recursive version"""
        self._resize_fmnks(m, n, np.max(k))
        for i in np.ravel(k):
            self._f(m, n, i)
        return self._fmnks[m, n, k] / special.binom(m + n, m)

    def pmf_iterative(self, k, m, n):
        """Probability mass function, iterative version"""
        fmnks = {}
        for i in np.ravel(k):
            fmnks = _mwu_f_iterative(m, n, i, fmnks)
        return np.array([fmnks[m, n, ki] for ki in k]) / special.binom(m + n, m)

    def cdf(self, k, m, n):
        """Cumulative distribution function"""
        pmfs = self.pmf(np.arange(0, np.max(k) + 1), m, n)
        cdfs = np.cumsum(pmfs)
        return cdfs[k]

    def sf(self, k, m, n):
        """Survival function"""
        k = m * n - k
        return self.cdf(k, m, n)

    def _resize_fmnks(self, m, n, k):
        """If necessary, expand the array that remembers PMF values"""
        shape_old = np.array(self._fmnks.shape)
        shape_new = np.array((m + 1, n + 1, k + 1))
        if np.any(shape_new > shape_old):
            shape = np.maximum(shape_old, shape_new)
            fmnks = -np.ones(shape)
            m0, n0, k0 = shape_old
            fmnks[:m0, :n0, :k0] = self._fmnks
            self._fmnks = fmnks

    def _f(self, m, n, k):
        """Recursive implementation of function of [3] Theorem 2.5"""
        if k < 0 or m < 0 or n < 0 or (k > m * n):
            return 0
        if self._fmnks[m, n, k] >= 0:
            return self._fmnks[m, n, k]
        if k == 0 and m >= 0 and (n >= 0):
            fmnk = 1
        else:
            fmnk = self._f(m - 1, n, k - n) + self._f(m, n - 1, k)
        self._fmnks[m, n, k] = fmnk
        return fmnk