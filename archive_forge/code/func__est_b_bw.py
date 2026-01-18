import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
def _est_b_bw(self):
    """
        Computes the (beta) coefficients and the bandwidths.

        Minimizes ``cv_loo`` with respect to ``b`` and ``bw``.
        """
    params0 = np.random.uniform(size=(self.k_linear + self.K,))
    b_bw = optimize.fmin(self.cv_loo, params0, disp=0)
    b = b_bw[0:self.k_linear]
    bw = b_bw[self.k_linear:]
    return (b, bw)