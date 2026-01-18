import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
class PHRegResults(base.LikelihoodModelResults):
    """
    Class to contain results of fitting a Cox proportional hazards
    survival model.

    PHregResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Attributes
    ----------
    model : class instance
        PHreg model instance that called fit.
    normalized_cov_params : ndarray
        The sampling covariance matrix of the estimates
    params : ndarray
        The coefficients of the fitted model.  Each coefficient is the
        log hazard ratio corresponding to a 1 unit difference in a
        single covariate while holding the other covariates fixed.
    bse : ndarray
        The standard errors of the fitted parameters.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    """

    def __init__(self, model, params, cov_params, scale=1.0, covariance_type='naive'):
        self.covariance_type = covariance_type
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        super().__init__(model, params, scale=1.0, normalized_cov_params=cov_params)

    @cache_readonly
    def standard_errors(self):
        """
        Returns the standard errors of the parameter estimates.
        """
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def bse(self):
        """
        Returns the standard errors of the parameter estimates.
        """
        return self.standard_errors

    def get_distribution(self):
        """
        Returns a scipy distribution object corresponding to the
        distribution of uncensored endog (duration) values for each
        case.

        Returns
        -------
        A list of objects of type scipy.stats.distributions.rv_discrete

        Notes
        -----
        The distributions are obtained from a simple discrete estimate
        of the survivor function that puts all mass on the observed
        failure times within a stratum.
        """
        return self.model.get_distribution(self.params)

    @Appender(_predict_docstring % {'params_doc': '', 'cov_params_doc': ''})
    def predict(self, endog=None, exog=None, strata=None, offset=None, transform=True, pred_type='lhr'):
        return super().predict(exog=exog, transform=transform, cov_params=self.cov_params(), endog=endog, strata=strata, offset=offset, pred_type=pred_type)

    def _group_stats(self, groups):
        """
        Descriptive statistics of the groups.
        """
        gsizes = np.unique(groups, return_counts=True)
        gsizes = gsizes[1]
        return (gsizes.min(), gsizes.max(), gsizes.mean(), len(gsizes))

    @cache_readonly
    def weighted_covariate_averages(self):
        """
        The average covariate values within the at-risk set at each
        event time point, weighted by hazard.
        """
        return self.model.weighted_covariate_averages(self.params)

    @cache_readonly
    def score_residuals(self):
        """
        A matrix containing the score residuals.
        """
        return self.model.score_residuals(self.params)

    @cache_readonly
    def baseline_cumulative_hazard(self):
        """
        A list (corresponding to the strata) containing the baseline
        cumulative hazard function evaluated at the event points.
        """
        return self.model.baseline_cumulative_hazard(self.params)

    @cache_readonly
    def baseline_cumulative_hazard_function(self):
        """
        A list (corresponding to the strata) containing function
        objects that calculate the cumulative hazard function.
        """
        return self.model.baseline_cumulative_hazard_function(self.params)

    @cache_readonly
    def schoenfeld_residuals(self):
        """
        A matrix containing the Schoenfeld residuals.

        Notes
        -----
        Schoenfeld residuals for censored observations are set to zero.
        """
        surv = self.model.surv
        w_avg = self.weighted_covariate_averages
        sch_resid = np.nan * np.ones(self.model.exog.shape, dtype=np.float64)
        for stx in range(surv.nstrat):
            uft = surv.ufailt[stx]
            exog_s = surv.exog_s[stx]
            time_s = surv.time_s[stx]
            strat_ix = surv.stratum_rows[stx]
            ii = np.searchsorted(uft, time_s)
            jj = np.flatnonzero(ii < len(uft))
            sch_resid[strat_ix[jj], :] = exog_s[jj, :] - w_avg[stx][ii[jj], :]
        jj = np.flatnonzero(self.model.status == 0)
        sch_resid[jj, :] = np.nan
        return sch_resid

    @cache_readonly
    def martingale_residuals(self):
        """
        The martingale residuals.
        """
        surv = self.model.surv
        mart_resid = np.nan * np.ones(len(self.model.endog), dtype=np.float64)
        cumhaz_f_list = self.baseline_cumulative_hazard_function
        for stx in range(surv.nstrat):
            cumhaz_f = cumhaz_f_list[stx]
            exog_s = surv.exog_s[stx]
            time_s = surv.time_s[stx]
            linpred = np.dot(exog_s, self.params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)
            ii = surv.stratum_rows[stx]
            chaz = cumhaz_f(time_s)
            mart_resid[ii] = self.model.status[ii] - e_linpred * chaz
        return mart_resid

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the proportional hazards regression results.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `x#` for ## in p the
            number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        float_format = '%8.3f'
        info = {}
        info['Model:'] = 'PH Reg'
        if yname is None:
            yname = self.model.endog_names
        info['Dependent variable:'] = yname
        info['Ties:'] = self.model.ties.capitalize()
        info['Sample size:'] = str(self.model.surv.n_obs)
        info['Num. events:'] = str(int(sum(self.model.status)))
        if self.model.groups is not None:
            mn, mx, avg, num = self._group_stats(self.model.groups)
            info['Num groups:'] = '%.0f' % num
            info['Min group size:'] = '%.0f' % mn
            info['Max group size:'] = '%.0f' % mx
            info['Avg group size:'] = '%.1f' % avg
        if self.model.strata is not None:
            mn, mx, avg, num = self._group_stats(self.model.strata)
            info['Num strata:'] = '%.0f' % num
            info['Min stratum size:'] = '%.0f' % mn
            info['Max stratum size:'] = '%.0f' % mx
            info['Avg stratum size:'] = '%.1f' % avg
        smry.add_dict(info, align='l', float_format=float_format)
        param = summary2.summary_params(self, alpha=alpha)
        param = param.rename(columns={'Coef.': 'log HR', 'Std.Err.': 'log HR SE'})
        param.insert(2, 'HR', np.exp(param['log HR']))
        a = '[%.3f' % (alpha / 2)
        param.loc[:, a] = np.exp(param.loc[:, a])
        a = '%.3f]' % (1 - alpha / 2)
        param.loc[:, a] = np.exp(param.loc[:, a])
        if xname is not None:
            param.index = xname
        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)
        smry.add_text('Confidence intervals are for the hazard ratios')
        dstrat = self.model.surv.nstrat_orig - self.model.surv.nstrat
        if dstrat > 0:
            if dstrat == 1:
                smry.add_text('1 stratum dropped for having no events')
            else:
                smry.add_text('%d strata dropped for having no events' % dstrat)
        if self.model.entry is not None:
            n_entry = sum(self.model.entry != 0)
            if n_entry == 1:
                smry.add_text('1 observation has a positive entry time')
            else:
                smry.add_text('%d observations have positive entry times' % n_entry)
        if self.model.groups is not None:
            smry.add_text('Standard errors account for dependence within groups')
        if hasattr(self, 'regularized'):
            smry.add_text('Standard errors do not account for the regularization')
        return smry