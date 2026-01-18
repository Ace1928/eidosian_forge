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
class MixedLMParams:
    """
    This class represents a parameter state for a mixed linear model.

    Parameters
    ----------
    k_fe : int
        The number of covariates with fixed effects.
    k_re : int
        The number of covariates with random coefficients (excluding
        variance components).
    k_vc : int
        The number of variance components parameters.

    Notes
    -----
    This object represents the parameter state for the model in which
    the scale parameter has been profiled out.
    """

    def __init__(self, k_fe, k_re, k_vc):
        self.k_fe = k_fe
        self.k_re = k_re
        self.k_re2 = k_re * (k_re + 1) // 2
        self.k_vc = k_vc
        self.k_tot = self.k_fe + self.k_re2 + self.k_vc
        self._ix = np.tril_indices(self.k_re)

    def from_packed(params, k_fe, k_re, use_sqrt, has_fe):
        """
        Create a MixedLMParams object from packed parameter vector.

        Parameters
        ----------
        params : array_like
            The mode parameters packed into a single vector.
        k_fe : int
            The number of covariates with fixed effects
        k_re : int
            The number of covariates with random effects (excluding
            variance components).
        use_sqrt : bool
            If True, the random effects covariance matrix is provided
            as its Cholesky factor, otherwise the lower triangle of
            the covariance matrix is stored.
        has_fe : bool
            If True, `params` contains fixed effects parameters.
            Otherwise, the fixed effects parameters are set to zero.

        Returns
        -------
        A MixedLMParams object.
        """
        k_re2 = int(k_re * (k_re + 1) / 2)
        if has_fe:
            k_vc = len(params) - k_fe - k_re2
        else:
            k_vc = len(params) - k_re2
        pa = MixedLMParams(k_fe, k_re, k_vc)
        cov_re = np.zeros((k_re, k_re))
        ix = pa._ix
        if has_fe:
            pa.fe_params = params[0:k_fe]
            cov_re[ix] = params[k_fe:k_fe + k_re2]
        else:
            pa.fe_params = np.zeros(k_fe)
            cov_re[ix] = params[0:k_re2]
        if use_sqrt:
            cov_re = np.dot(cov_re, cov_re.T)
        else:
            cov_re = cov_re + cov_re.T - np.diag(np.diag(cov_re))
        pa.cov_re = cov_re
        if k_vc > 0:
            if use_sqrt:
                pa.vcomp = params[-k_vc:] ** 2
            else:
                pa.vcomp = params[-k_vc:]
        else:
            pa.vcomp = np.array([])
        return pa
    from_packed = staticmethod(from_packed)

    def from_components(fe_params=None, cov_re=None, cov_re_sqrt=None, vcomp=None):
        """
        Create a MixedLMParams object from each parameter component.

        Parameters
        ----------
        fe_params : array_like
            The fixed effects parameter (a 1-dimensional array).  If
            None, there are no fixed effects.
        cov_re : array_like
            The random effects covariance matrix (a square, symmetric
            2-dimensional array).
        cov_re_sqrt : array_like
            The Cholesky (lower triangular) square root of the random
            effects covariance matrix.
        vcomp : array_like
            The variance component parameters.  If None, there are no
            variance components.

        Returns
        -------
        A MixedLMParams object.
        """
        if vcomp is None:
            vcomp = np.empty(0)
        if fe_params is None:
            fe_params = np.empty(0)
        if cov_re is None and cov_re_sqrt is None:
            cov_re = np.empty((0, 0))
        k_fe = len(fe_params)
        k_vc = len(vcomp)
        k_re = cov_re.shape[0] if cov_re is not None else cov_re_sqrt.shape[0]
        pa = MixedLMParams(k_fe, k_re, k_vc)
        pa.fe_params = fe_params
        if cov_re_sqrt is not None:
            pa.cov_re = np.dot(cov_re_sqrt, cov_re_sqrt.T)
        elif cov_re is not None:
            pa.cov_re = cov_re
        pa.vcomp = vcomp
        return pa
    from_components = staticmethod(from_components)

    def copy(self):
        """
        Returns a copy of the object.
        """
        obj = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
        obj.fe_params = self.fe_params.copy()
        obj.cov_re = self.cov_re.copy()
        obj.vcomp = self.vcomp.copy()
        return obj

    def get_packed(self, use_sqrt, has_fe=False):
        """
        Return the model parameters packed into a single vector.

        Parameters
        ----------
        use_sqrt : bool
            If True, the Cholesky square root of `cov_re` is
            included in the packed result.  Otherwise the
            lower triangle of `cov_re` is included.
        has_fe : bool
            If True, the fixed effects parameters are included
            in the packed result, otherwise they are omitted.
        """
        if self.k_re > 0:
            if use_sqrt:
                try:
                    L = np.linalg.cholesky(self.cov_re)
                except np.linalg.LinAlgError:
                    L = np.diag(np.sqrt(np.diag(self.cov_re)))
                cpa = L[self._ix]
            else:
                cpa = self.cov_re[self._ix]
        else:
            cpa = np.zeros(0)
        if use_sqrt:
            vcomp = np.sqrt(self.vcomp)
        else:
            vcomp = self.vcomp
        if has_fe:
            pa = np.concatenate((self.fe_params, cpa, vcomp))
        else:
            pa = np.concatenate((cpa, vcomp))
        return pa