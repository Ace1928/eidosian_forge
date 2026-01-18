import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
def _compute_test_stat(self, u):
    n = np.shape(u)[0]
    XLOO = LeaveOneOut(self.exog)
    uLOO = LeaveOneOut(u[:, None]).__iter__()
    ival = 0
    S2 = 0
    for i, X_not_i in enumerate(XLOO):
        u_j = next(uLOO)
        u_j = np.squeeze(u_j)
        K = gpke(self.bw, data=-X_not_i, data_predict=-self.exog[i, :], var_type=self.var_type, tosum=False)
        f_i = u[i] * u_j * K
        assert u_j.shape == K.shape
        ival += f_i.sum()
        S2 += (f_i ** 2).sum()
        assert np.size(ival) == 1
        assert np.size(S2) == 1
    ival *= 1.0 / (n * (n - 1))
    ix_cont = _get_type_pos(self.var_type)[0]
    hp = self.bw[ix_cont].prod()
    S2 *= 2 * hp / (n * (n - 1))
    T = n * ival * np.sqrt(hp / S2)
    return T