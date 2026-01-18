import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def _fit_pa(self, maxiter=50, tol=1e-08):
    """
        Extract factors using the iterative principal axis method

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations for communality estimation
        tol : float
            If `norm(communality - last_communality)  < tolerance`,
            estimation stops

        Returns
        -------
        results : FactorResults instance
        """
    R = self.corr.copy()
    self.n_comp = matrix_rank(R)
    if self.n_factor > self.n_comp:
        raise ValueError('n_factor must be smaller or equal to the rank of endog! %d > %d' % (self.n_factor, self.n_comp))
    if maxiter <= 0:
        raise ValueError('n_max_iter must be larger than 0! %d < 0' % maxiter)
    if tol <= 0 or tol > 0.01:
        raise ValueError('tolerance must be larger than 0 and smaller than 0.01! Got %f instead' % tol)
    if self.smc:
        c = 1 - 1 / np.diag(inv(R))
    else:
        c = np.ones(len(R))
    eigenvals = None
    for i in range(maxiter):
        for j in range(len(R)):
            R[j, j] = c[j]
        L, V = eigh(R, UPLO='U')
        c_last = np.array(c)
        ind = np.argsort(L)
        ind = ind[::-1]
        L = L[ind]
        n_pos = (L > 0).sum()
        V = V[:, ind]
        eigenvals = np.array(L)
        n = np.min([n_pos, self.n_factor])
        sL = np.diag(np.sqrt(L[:n]))
        V = V[:, :n]
        A = V.dot(sL)
        c = np.power(A, 2).sum(axis=1)
        if norm(c_last - c) < tol:
            break
    self.eigenvals = eigenvals
    self.communality = c
    self.uniqueness = 1 - c
    self.loadings = A
    return FactorResults(self)