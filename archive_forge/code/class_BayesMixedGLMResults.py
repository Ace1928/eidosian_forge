from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
class BayesMixedGLMResults:
    """
    Class to hold results from a Bayesian estimation of a Mixed GLM model.

    Attributes
    ----------
    fe_mean : array_like
        Posterior mean of the fixed effects coefficients.
    fe_sd : array_like
        Posterior standard deviation of the fixed effects coefficients
    vcp_mean : array_like
        Posterior mean of the logged variance component standard
        deviations.
    vcp_sd : array_like
        Posterior standard deviation of the logged variance component
        standard deviations.
    vc_mean : array_like
        Posterior mean of the random coefficients
    vc_sd : array_like
        Posterior standard deviation of the random coefficients
    """

    def __init__(self, model, params, cov_params, optim_retvals=None):
        self.model = model
        self.params = params
        self._cov_params = cov_params
        self.optim_retvals = optim_retvals
        self.fe_mean, self.vcp_mean, self.vc_mean = model._unpack(params)
        if cov_params.ndim == 2:
            cp = np.diag(cov_params)
        else:
            cp = cov_params
        self.fe_sd, self.vcp_sd, self.vc_sd = model._unpack(cp)
        self.fe_sd = np.sqrt(self.fe_sd)
        self.vcp_sd = np.sqrt(self.vcp_sd)
        self.vc_sd = np.sqrt(self.vc_sd)

    def cov_params(self):
        if hasattr(self.model.data, 'frame'):
            na = self.model.fep_names + self.model.vcp_names + self.model.vc_names
            if self._cov_params.ndim == 2:
                return pd.DataFrame(self._cov_params, index=na, columns=na)
            else:
                return pd.Series(self._cov_params, index=na)
        return self._cov_params

    def summary(self):
        df = pd.DataFrame()
        m = self.model.k_fep + self.model.k_vcp
        df['Type'] = ['M' for k in range(self.model.k_fep)] + ['V' for k in range(self.model.k_vcp)]
        df['Post. Mean'] = self.params[0:m]
        if self._cov_params.ndim == 2:
            v = np.diag(self._cov_params)[0:m]
            df['Post. SD'] = np.sqrt(v)
        else:
            df['Post. SD'] = np.sqrt(self._cov_params[0:m])
        df['SD'] = np.exp(df['Post. Mean'])
        df['SD (LB)'] = np.exp(df['Post. Mean'] - 2 * df['Post. SD'])
        df['SD (UB)'] = np.exp(df['Post. Mean'] + 2 * df['Post. SD'])
        df['SD'] = ['%.3f' % x for x in df.SD]
        df['SD (LB)'] = ['%.3f' % x for x in df['SD (LB)']]
        df['SD (UB)'] = ['%.3f' % x for x in df['SD (UB)']]
        df.loc[df.index < self.model.k_fep, 'SD'] = ''
        df.loc[df.index < self.model.k_fep, 'SD (LB)'] = ''
        df.loc[df.index < self.model.k_fep, 'SD (UB)'] = ''
        df.index = self.model.fep_names + self.model.vcp_names
        summ = summary2.Summary()
        summ.add_title(self.model.family.__class__.__name__ + ' Mixed GLM Results')
        summ.add_df(df)
        summ.add_text('Parameter types are mean structure (M) and variance structure (V)')
        summ.add_text('Variance parameters are modeled as log standard deviations')
        return summ

    def random_effects(self, term=None):
        """
        Posterior mean and standard deviation of random effects.

        Parameters
        ----------
        term : int or None
            If None, results for all random effects are returned.  If
            an integer, returns results for a given set of random
            effects.  The value of `term` refers to an element of the
            `ident` vector, or to a position in the `vc_formulas`
            list.

        Returns
        -------
        Data frame of posterior means and posterior standard
        deviations of random effects.
        """
        z = self.vc_mean
        s = self.vc_sd
        na = self.model.vc_names
        if term is not None:
            termix = self.model.vcp_names.index(term)
            ii = np.flatnonzero(self.model.ident == termix)
            z = z[ii]
            s = s[ii]
            na = [na[i] for i in ii]
        x = pd.DataFrame({'Mean': z, 'SD': s})
        if na is not None:
            x.index = na
        return x

    def predict(self, exog=None, linear=False):
        """
        Return predicted values for the mean structure.

        Parameters
        ----------
        exog : array_like
            The design matrix for the mean structure.  If None,
            use the model's design matrix.
        linear : bool
            If True, returns the linear predictor, otherwise
            transform the linear predictor using the link function.

        Returns
        -------
        A one-dimensional array of fitted values.
        """
        return self.model.predict(self.params, exog, linear)