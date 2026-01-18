import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _cv_ml(self):
    """
        Returns the cross validation maximum likelihood bandwidth parameter.

        Notes
        -----
        For more details see p.16, 18, 27 in Ref. [1] (see module docstring).

        Returns the bandwidth estimate that maximizes the leave-out-out
        likelihood.  The leave-one-out log likelihood function is:

        .. math:: \\ln L=\\sum_{i=1}^{n}\\ln f_{-i}(X_{i})

        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\\frac{1}{(n-1)h}
                        \\sum_{j=1,j\\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the Generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j})=\\prod_{s=1}^
                        {q}h_{s}^{-1}k\\left(\\frac{X_{is}-X_{js}}{h_{s}}\\right)
        """
    h0 = self._normal_reference()
    bw = optimize.fmin(self.loo_likelihood, x0=h0, args=(np.log,), maxiter=1000.0, maxfun=1000.0, disp=0, xtol=0.001)
    bw = self._set_bw_bounds(bw)
    return bw