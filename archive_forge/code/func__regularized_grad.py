import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _regularized_grad(self, A):
    p = self.k_vars
    ndim = self.ndim
    covx = self._covx
    n_slice = self.n_slice
    mn = self._slice_means
    ph = self._slice_props
    A = A.reshape((p, ndim))
    gr = 2 * np.dot(self.pen_mat.T, np.dot(self.pen_mat, A))
    A = A.reshape((p, ndim))
    covxa = np.dot(covx, A)
    covx2a = np.dot(covx, covxa)
    Q = np.dot(covxa.T, covxa)
    Qi = np.linalg.inv(Q)
    jm = np.zeros((p, ndim))
    qcv = np.linalg.solve(Q, covxa.T)
    ft = [None] * (p * ndim)
    for q in range(p):
        for r in range(ndim):
            jm *= 0
            jm[q, r] = 1
            umat = np.dot(covx2a.T, jm)
            umat += umat.T
            umat = -np.dot(Qi, np.dot(umat, Qi))
            fmat = np.dot(np.dot(covx, jm), qcv)
            fmat += np.dot(covxa, np.dot(umat, covxa.T))
            fmat += np.dot(covxa, np.linalg.solve(Q, np.dot(jm.T, covx)))
            ft[q * ndim + r] = fmat
    ch = np.linalg.solve(Q, np.dot(covxa.T, mn.T))
    cu = mn - np.dot(covxa, ch).T
    for i in range(n_slice):
        u = cu[i, :]
        v = mn[i, :]
        for q in range(p):
            for r in range(ndim):
                f = np.dot(u, np.dot(ft[q * ndim + r], v))
                gr[q, r] -= 2 * ph[i] * f
    return gr.ravel()