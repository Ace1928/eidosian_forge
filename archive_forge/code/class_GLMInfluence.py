import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
class GLMInfluence(MLEInfluence):
    """Influence and outlier measures (experimental)

    This uses partly formulas specific to GLM, specifically cooks_distance
    is based on the hessian, i.e. observed or expected information matrix and
    not on cov_params, in contrast to MLEInfluence.
    Standardization for changes in parameters, in fittedvalues and in
    the linear predictor are based on cov_params.

    Parameters
    ----------
    results : instance of results class
        This only works for model and results classes that have the necessary
        helper methods.
    other arguments are only to override default behavior and are used instead
    of the corresponding attribute of the results class.
    By default resid_pearson is used as resid.

    Attributes
    ----------
    dbetas
        change in parameters divided by the standard error of parameters from
        the full model results, ``bse``.
    d_fittedvalues_scaled
        same as d_fittedvalues but scaled by the standard errors of a
        predicted mean of the response.
    d_linpred
        local change in linear prediction.
    d_linpred_scale
        local change in linear prediction scaled by the standard errors for
        the prediction based on cov_params.

    Notes
    -----
    This has not yet been tested for correctness when offset or exposure
    are used, although they should be supported by the code.

    Some GLM specific measures like d_deviance are still missing.

    Computing an explicit leave-one-observation-out (LOOO) loop is included
    but no influence measures are currently computed from it.
    """

    @cache_readonly
    def hat_matrix_diag(self):
        """
        Diagonal of the hat_matrix for GLM

        Notes
        -----
        This returns the diagonal of the hat matrix that was provided as
        argument to GLMInfluence or computes it using the results method
        `get_hat_matrix`.
        """
        if hasattr(self, '_hat_matrix_diag'):
            return self._hat_matrix_diag
        else:
            return self.results.get_hat_matrix()

    @cache_readonly
    def d_params(self):
        """Change in parameter estimates

        Notes
        -----
        This uses one-step approximation of the parameter change to deleting
        one observation.
        """
        beta_i = np.linalg.pinv(self.exog) * self.resid_studentized
        beta_i /= np.sqrt(1 - self.hat_matrix_diag)
        return beta_i.T

    @cache_readonly
    def resid_studentized(self):
        """
        Internally studentized pearson residuals

        Notes
        -----
        residuals / sqrt( scale * (1 - hii))

        where residuals are those provided to GLMInfluence which are
        pearson residuals by default, and
        hii is the diagonal of the hat matrix.
        """
        return super().resid_studentized

    @cache_readonly
    def cooks_distance(self):
        """Cook's distance

        Notes
        -----
        Based on one step approximation using resid_studentized and
        hat_matrix_diag for the computation.

        Cook's distance divides by the number of explanatory variables.

        Computed using formulas for GLM and does not use results.cov_params.
        It includes p-values based on the F-distribution which are only
        approximate outside of linear Gaussian models.
        """
        hii = self.hat_matrix_diag
        cooks_d2 = self.resid_studentized ** 2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)
        from scipy import stats
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)
        return (cooks_d2, pvals)

    @property
    def d_linpred(self):
        """
        Change in linear prediction

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``.
        """
        exog = self.results.model.exog
        return (exog * self.d_params).sum(1)

    @property
    def d_linpred_scaled(self):
        """
        Change in linpred scaled by standard errors

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``, and divides by the standard errors
        for linpred provided by results.get_prediction.
        """
        return self.d_linpred / self._get_prediction.linpred.se

    @property
    def _fittedvalues_one(self):
        """experimental code
        """
        warnings.warn('this ignores offset and exposure', UserWarning)
        exog = self.results.model.exog
        fitted = np.array([self.results.model.predict(pi, exog[i]) for i, pi in enumerate(self.params_one)])
        return fitted.squeeze()

    @property
    def _diff_fittedvalues_one(self):
        """experimental code
        """
        return self.results.predict() - self._fittedvalues_one

    @cache_readonly
    def _res_looo(self):
        """collect required results from the LOOO loop

        all results will be attached.
        currently only 'params', 'mse_resid', 'det_cov_params' are stored

        Reestimates the model with endog and exog dropping one observation
        at a time

        This uses a nobs loop, only attributes of the results instance are
        stored.

        Warning: This will need refactoring and API changes to be able to
        add options.
        """
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut
        get_det_cov_params = lambda res: np.linalg.det(res.cov_params())
        endog = self.results.model.endog
        exog = self.results.model.exog
        init_kwds = self.results.model._get_init_kwds()
        freq_weights = init_kwds.pop('freq_weights')
        var_weights = init_kwds.pop('var_weights')
        offset = offset_ = init_kwds.pop('offset')
        exposure = exposure_ = init_kwds.pop('exposure')
        n_trials = init_kwds.pop('n_trials', None)
        if hasattr(init_kwds['family'], 'initialize'):
            is_binomial = True
        else:
            is_binomial = False
        params = np.zeros(exog.shape, dtype=float)
        scale = np.zeros(endog.shape, dtype=float)
        det_cov_params = np.zeros(endog.shape, dtype=float)
        cv_iter = LeaveOneOut(self.nobs)
        for inidx, outidx in cv_iter:
            if offset is not None:
                offset_ = offset[inidx]
            if exposure is not None:
                exposure_ = exposure[inidx]
            if n_trials is not None:
                init_kwds['n_trials'] = n_trials[inidx]
            mod_i = self.model_class(endog[inidx], exog[inidx], offset=offset_, exposure=exposure_, freq_weights=freq_weights[inidx], var_weights=var_weights[inidx], **init_kwds)
            if is_binomial:
                mod_i.family.n = init_kwds['n_trials']
            res_i = mod_i.fit(start_params=self.results.params, method='newton')
            params[outidx] = res_i.params.copy()
            scale[outidx] = res_i.scale
            det_cov_params[outidx] = get_det_cov_params(res_i)
        return dict(params=params, scale=scale, mse_resid=scale, det_cov_params=det_cov_params)