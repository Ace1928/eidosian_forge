import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def _orth_cov(self):
    Ik = np.eye(self.neqs)
    PIk = np.kron(self.P.T, Ik)
    H = self.H
    covs = self._empty_covm(self.periods + 1)
    for i in range(self.periods + 1):
        if i == 0:
            apiece = 0
        else:
            Ci = np.dot(PIk, self.G[i - 1])
            apiece = Ci @ self.cov_a @ Ci.T
        Cibar = np.dot(np.kron(Ik, self.irfs[i]), H)
        bpiece = Cibar @ self.cov_sig @ Cibar.T / self.T
        covs[i] = apiece + bpiece
    return covs