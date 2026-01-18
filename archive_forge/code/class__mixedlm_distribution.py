import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class _mixedlm_distribution:
    """
    A private class for simulating data from a given mixed linear model.

    Parameters
    ----------
    model : MixedLM instance
        A mixed linear model
    params : array_like
        A parameter vector defining a mixed linear model.  See
        notes for more information.
    scale : scalar
        The unexplained variance
    exog : array_like
        An array of fixed effect covariates.  If None, model.exog
        is used.

    Notes
    -----
    The params array is a vector containing fixed effects parameters,
    random effects parameters, and variance component parameters, in
    that order.  The lower triangle of the random effects covariance
    matrix is stored.  The random effects and variance components
    parameters are divided by the scale parameter.

    This class is used in Mediation, and possibly elsewhere.
    """

    def __init__(self, model, params, scale, exog):
        self.model = model
        self.exog = exog if exog is not None else model.exog
        po = MixedLMParams.from_packed(params, model.k_fe, model.k_re, False, True)
        self.fe_params = po.fe_params
        self.cov_re = scale * po.cov_re
        self.vcomp = scale * po.vcomp
        self.scale = scale
        group_idx = np.zeros(model.nobs, dtype=int)
        for k, g in enumerate(model.group_labels):
            group_idx[model.row_indices[g]] = k
        self.group_idx = group_idx

    def rvs(self, n):
        """
        Return a vector of simulated values from a mixed linear
        model.

        The parameter n is ignored, but required by the interface
        """
        model = self.model
        y = np.dot(self.exog, self.fe_params)
        u = np.random.normal(size=(model.n_groups, model.k_re))
        u = np.dot(u, np.linalg.cholesky(self.cov_re).T)
        y += (u[self.group_idx, :] * model.exog_re).sum(1)
        for j, _ in enumerate(model.exog_vc.names):
            ex = model.exog_vc.mats[j]
            v = self.vcomp[j]
            for i, g in enumerate(model.group_labels):
                exg = ex[i]
                ii = model.row_indices[g]
                u = np.random.normal(size=exg.shape[1])
                y[ii] += np.sqrt(v) * np.dot(exg, u)
        y += np.sqrt(self.scale) * np.random.normal(size=len(y))
        return y