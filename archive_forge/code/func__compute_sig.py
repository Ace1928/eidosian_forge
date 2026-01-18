import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
def _compute_sig(self):
    Y = self.endog
    X = self.exog
    b = self.estimator(Y, X)
    m = self.fform(X, b)
    n = np.shape(X)[0]
    resid = Y - m
    resid = resid - np.mean(resid)
    self.test_stat = self._compute_test_stat(resid)
    sqrt5 = np.sqrt(5.0)
    fct1 = (1 - sqrt5) / 2.0
    fct2 = (1 + sqrt5) / 2.0
    u1 = fct1 * resid
    u2 = fct2 * resid
    r = fct2 / sqrt5
    I_dist = np.empty((self.nboot, 1))
    for j in range(self.nboot):
        u_boot = u2.copy()
        prob = np.random.uniform(0, 1, size=(n,))
        ind = prob < r
        u_boot[ind] = u1[ind]
        Y_boot = m + u_boot
        b_hat = self.estimator(Y_boot, X)
        m_hat = self.fform(X, b_hat)
        u_boot_hat = Y_boot - m_hat
        I_dist[j] = self._compute_test_stat(u_boot_hat)
    self.boots_results = I_dist
    sig = 'Not Significant'
    if self.test_stat > mquantiles(I_dist, 0.9):
        sig = '*'
    if self.test_stat > mquantiles(I_dist, 0.95):
        sig = '**'
    if self.test_stat > mquantiles(I_dist, 0.99):
        sig = '***'
    return sig