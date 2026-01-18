import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def _est_loc_constant(self, bw, endog, exog, data_predict):
    """
        Local constant estimator of g(x) in the regression
        y = g(x) + e

        Parameters
        ----------
        bw : array_like
            Array of bandwidth value(s).
        endog : 1D array_like
            The dependent variable.
        exog : 1D or 2D array_like
            The independent variable(s).
        data_predict : 1D or 2D array_like
            The point(s) at which the density is estimated.

        Returns
        -------
        G : ndarray
            The value of the conditional mean at `data_predict`.
        B_x : ndarray
            The marginal effects.
        """
    ker_x = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False)
    ker_x = np.reshape(ker_x, np.shape(endog))
    G_numer = (ker_x * endog).sum(axis=0)
    G_denom = ker_x.sum(axis=0)
    G = G_numer / G_denom
    nobs = exog.shape[0]
    f_x = G_denom / float(nobs)
    ker_xc = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype='d_gaussian', tosum=False)
    ker_xc = ker_xc[:, np.newaxis]
    d_mx = -(endog * ker_xc).sum(axis=0) / float(nobs)
    d_fx = -ker_xc.sum(axis=0) / float(nobs)
    B_x = d_mx / f_x - G * d_fx / f_x
    B_x = (G_numer * d_fx - G_denom * d_mx) / G_denom ** 2
    return (G, B_x)