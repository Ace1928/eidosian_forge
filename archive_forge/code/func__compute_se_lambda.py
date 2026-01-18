import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def _compute_se_lambda(self, Y, X):
    """
        Calculates the SE of lambda by nested resampling
        Used to pivot the statistic.
        Bootstrapping works better with estimating pivotal statistics
        but slows down computation significantly.
        """
    n = np.shape(Y)[0]
    lam = np.empty(shape=(self.nres,))
    for i in range(self.nres):
        ind = np.random.randint(0, n, size=(n, 1))
        Y1 = Y[ind, 0]
        X1 = X[ind, :]
        lam[i] = self._compute_lambda(Y1, X1)
    se_lambda = np.std(lam)
    return se_lambda