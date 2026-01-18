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
class MixedLMResults(base.LikelihoodModelResults, base.ResultMixin):
    """
    Class to contain results of fitting a linear mixed effects model.

    MixedLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Attributes
    ----------
    model : class instance
        Pointer to MixedLM model instance that called fit.
    normalized_cov_params : ndarray
        The sampling covariance matrix of the estimates
    params : ndarray
        A packed parameter vector for the profile parameterization.
        The first `k_fe` elements are the estimated fixed effects
        coefficients.  The remaining elements are the estimated
        variance parameters.  The variance parameters are all divided
        by `scale` and are not the variance parameters shown
        in the summary.
    fe_params : ndarray
        The fitted fixed-effects coefficients
    cov_re : ndarray
        The fitted random-effects covariance matrix
    bse_fe : ndarray
        The standard errors of the fitted fixed effects coefficients
    bse_re : ndarray
        The standard errors of the fitted random effects covariance
        matrix and variance components.  The first `k_re * (k_re + 1)`
        parameters are the standard errors for the lower triangle of
        `cov_re`, the remaining elements are the standard errors for
        the variance components.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    """

    def __init__(self, model, params, cov_params):
        super().__init__(model, params, normalized_cov_params=cov_params)
        self.nobs = self.model.nobs
        self.df_resid = self.nobs - np.linalg.matrix_rank(self.model.exog)

    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values for the model.

        The fitted values reflect the mean structure specified by the
        fixed effects and the predicted random effects.
        """
        fit = np.dot(self.model.exog, self.fe_params)
        re = self.random_effects
        for group_ix, group in enumerate(self.model.group_labels):
            ix = self.model.row_indices[group]
            mat = []
            if self.model.exog_re_li is not None:
                mat.append(self.model.exog_re_li[group_ix])
            for j in range(self.k_vc):
                mat.append(self.model.exog_vc.mats[j][group_ix])
            mat = np.concatenate(mat, axis=1)
            fit[ix] += np.dot(mat, re[group])
        return fit

    @cache_readonly
    def resid(self):
        """
        Returns the residuals for the model.

        The residuals reflect the mean structure specified by the
        fixed effects and the predicted random effects.
        """
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def bse_fe(self):
        """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(np.diag(self.cov_params())[0:p])

    @cache_readonly
    def bse_re(self):
        """
        Returns the standard errors of the variance parameters.

        The first `k_re x (k_re + 1)` elements of the returned array
        are the standard errors of the lower triangle of `cov_re`.
        The remaining elements are the standard errors of the variance
        components.

        Note that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        or p-values if used in the usual way.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(self.scale * np.diag(self.cov_params())[p:])

    def _expand_re_names(self, group_ix):
        names = list(self.model.data.exog_re_names)
        for j, v in enumerate(self.model.exog_vc.names):
            vg = self.model.exog_vc.colnames[j][group_ix]
            na = ['{}[{}]'.format(v, s) for s in vg]
            names.extend(na)
        return names

    @cache_readonly
    def random_effects(self):
        """
        The conditional means of random effects given the data.

        Returns
        -------
        random_effects : dict
            A dictionary mapping the distinct `group` values to the
            conditional means of the random effects for the group
            given the data.
        """
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            raise ValueError('Cannot predict random effects from ' + 'singular covariance structure.')
        vcomp = self.vcomp
        k_re = self.k_re
        ranef_dict = {}
        for group_ix, group in enumerate(self.model.group_labels):
            endog = self.model.endog_li[group_ix]
            exog = self.model.exog_li[group_ix]
            ex_r = self.model._aex_r[group_ix]
            ex2_r = self.model._aex_r2[group_ix]
            vc_var = self.model._expand_vcomp(vcomp, group_ix)
            resid = endog
            if self.k_fe > 0:
                expval = np.dot(exog, self.fe_params)
                resid = resid - expval
            solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            vir = solver(resid)
            xtvir = _dot(ex_r.T, vir)
            xtvir[0:k_re] = np.dot(self.cov_re, xtvir[0:k_re])
            xtvir[k_re:] *= vc_var
            ranef_dict[group] = pd.Series(xtvir, index=self._expand_re_names(group_ix))
        return ranef_dict

    @cache_readonly
    def random_effects_cov(self):
        """
        Returns the conditional covariance matrix of the random
        effects for each group given the data.

        Returns
        -------
        random_effects_cov : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None
        vcomp = self.vcomp
        ranef_dict = {}
        for group_ix in range(self.model.n_groups):
            ex_r = self.model._aex_r[group_ix]
            ex2_r = self.model._aex_r2[group_ix]
            label = self.model.group_labels[group_ix]
            vc_var = self.model._expand_vcomp(vcomp, group_ix)
            solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            n = ex_r.shape[0]
            m = self.cov_re.shape[0]
            mat1 = np.empty((n, m + len(vc_var)))
            mat1[:, 0:m] = np.dot(ex_r[:, 0:m], self.cov_re)
            mat1[:, m:] = np.dot(ex_r[:, m:], np.diag(vc_var))
            mat2 = solver(mat1)
            mat2 = np.dot(mat1.T, mat2)
            v = -mat2
            v[0:m, 0:m] += self.cov_re
            ix = np.arange(m, v.shape[0])
            v[ix, ix] += vc_var
            na = self._expand_re_names(group_ix)
            v = pd.DataFrame(v, index=na, columns=na)
            ranef_dict[label] = v
        return ranef_dict

    def t_test(self, r_matrix, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q

        Parameters
        ----------
        r_matrix : array_like
            If an array is given, a p x k 2d array or length k 1d
            array specifying the linear restrictions. It is assumed
            that the linear combination is equal to zero.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.
        use_t : bool, optional
            If use_t is None, then the default of the model is used.
            If use_t is True, then the p-values are based on the t
            distribution.
            If use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.
        """
        if r_matrix.shape[1] != self.k_fe:
            raise ValueError('r_matrix for t-test should have %d columns' % self.k_fe)
        d = self.k_re2 + self.k_vc
        z0 = np.zeros((r_matrix.shape[0], d))
        r_matrix = np.concatenate((r_matrix, z0), axis=1)
        tst_rslt = super().t_test(r_matrix, use_t=use_t)
        return tst_rslt

    def summary(self, yname=None, xname_fe=None, xname_re=None, title=None, alpha=0.05):
        """
        Summarize the mixed model regression results.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname_fe : list[str], optional
            Fixed effects covariate names
        xname_re : list[str], optional
            Random effects covariate names
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
        info = {}
        info['Model:'] = 'MixedLM'
        if yname is None:
            yname = self.model.endog_names
        param_names = self.model.data.param_names[:]
        k_fe_params = len(self.fe_params)
        k_re_params = len(param_names) - len(self.fe_params)
        if xname_fe is not None:
            if len(xname_fe) != k_fe_params:
                msg = 'xname_fe should be a list of length %d' % k_fe_params
                raise ValueError(msg)
            param_names[:k_fe_params] = xname_fe
        if xname_re is not None:
            if len(xname_re) != k_re_params:
                msg = 'xname_re should be a list of length %d' % k_re_params
                raise ValueError(msg)
            param_names[k_fe_params:] = xname_re
        info['No. Observations:'] = str(self.model.n_totobs)
        info['No. Groups:'] = str(self.model.n_groups)
        gs = np.array([len(x) for x in self.model.endog_li])
        info['Min. group size:'] = '%.0f' % min(gs)
        info['Max. group size:'] = '%.0f' % max(gs)
        info['Mean group size:'] = '%.1f' % np.mean(gs)
        info['Dependent Variable:'] = yname
        info['Method:'] = self.method
        info['Scale:'] = self.scale
        info['Log-Likelihood:'] = self.llf
        info['Converged:'] = 'Yes' if self.converged else 'No'
        smry.add_dict(info)
        smry.add_title('Mixed Linear Model Regression Results')
        float_fmt = '%.3f'
        sdf = np.nan * np.ones((self.k_fe + self.k_re2 + self.k_vc, 6))
        sdf[0:self.k_fe, 0] = self.fe_params
        sdf[0:self.k_fe, 1] = np.sqrt(np.diag(self.cov_params()[0:self.k_fe]))
        sdf[0:self.k_fe, 2] = sdf[0:self.k_fe, 0] / sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 3] = 2 * norm.cdf(-np.abs(sdf[0:self.k_fe, 2]))
        qm = -norm.ppf(alpha / 2)
        sdf[0:self.k_fe, 4] = sdf[0:self.k_fe, 0] - qm * sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 5] = sdf[0:self.k_fe, 0] + qm * sdf[0:self.k_fe, 1]
        jj = self.k_fe
        for i in range(self.k_re):
            for j in range(i + 1):
                sdf[jj, 0] = self.cov_re[i, j]
                sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
                jj += 1
        for i in range(self.k_vc):
            sdf[jj, 0] = self.vcomp[i]
            sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
            jj += 1
        sdf = pd.DataFrame(index=param_names, data=sdf)
        sdf.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
        for col in sdf.columns:
            sdf[col] = [float_fmt % x if np.isfinite(x) else '' for x in sdf[col]]
        smry.add_df(sdf, align='r')
        return smry

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params_object, profile_fe=False)

    @cache_readonly
    def aic(self):
        """Akaike information criterion"""
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * (self.llf - df)

    @cache_readonly
    def bic(self):
        """Bayesian information criterion"""
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * self.llf + np.log(self.nobs) * df

    def profile_re(self, re_ix, vtype, num_low=5, dist_low=1.0, num_high=5, dist_high=1.0, **fit_kwargs):
        """
        Profile-likelihood inference for variance parameters.

        Parameters
        ----------
        re_ix : int
            If vtype is `re`, this value is the index of the variance
            parameter for which to construct a profile likelihood.  If
            `vtype` is 'vc' then `re_ix` is the name of the variance
            parameter to be profiled.
        vtype : str
            Either 're' or 'vc', depending on whether the profile
            analysis is for a random effect or a variance component.
        num_low : int
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : int
            The number of points at which to calculate the likelihood
            above the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        An array with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.

        Notes
        -----
        Only variance parameters can be profiled.
        """
        pmodel = self.model
        k_fe = pmodel.k_fe
        k_re = pmodel.k_re
        k_vc = pmodel.k_vc
        endog, exog = (pmodel.endog, pmodel.exog)
        if vtype == 're':
            ix = np.arange(k_re)
            ix[0] = re_ix
            ix[re_ix] = 0
            exog_re = pmodel.exog_re.copy()[:, ix]
            params = self.params_object.copy()
            cov_re_unscaled = params.cov_re
            cov_re_unscaled = cov_re_unscaled[np.ix_(ix, ix)]
            params.cov_re = cov_re_unscaled
            ru0 = cov_re_unscaled[0, 0]
            cov_re = self.scale * cov_re_unscaled
            low = (cov_re[0, 0] - dist_low) / self.scale
            high = (cov_re[0, 0] + dist_high) / self.scale
        elif vtype == 'vc':
            re_ix = self.model.exog_vc.names.index(re_ix)
            params = self.params_object.copy()
            vcomp = self.vcomp
            low = (vcomp[re_ix] - dist_low) / self.scale
            high = (vcomp[re_ix] + dist_high) / self.scale
            ru0 = vcomp[re_ix] / self.scale
        if low <= 0:
            raise ValueError('dist_low is too large and would result in a negative variance. Try a smaller value.')
        left = np.linspace(low, ru0, num_low + 1)
        right = np.linspace(ru0, high, num_high + 1)[1:]
        rvalues = np.concatenate((left, right))
        free = MixedLMParams(k_fe, k_re, k_vc)
        if self.freepat is None:
            free.fe_params = np.ones(k_fe)
            vcomp = np.ones(k_vc)
            mat = np.ones((k_re, k_re))
        else:
            free.fe_params = self.freepat.fe_params
            vcomp = self.freepat.vcomp
            mat = self.freepat.cov_re
            if vtype == 're':
                mat = mat[np.ix_(ix, ix)]
        if vtype == 're':
            mat[0, 0] = 0
        else:
            vcomp[re_ix] = 0
        free.cov_re = mat
        free.vcomp = vcomp
        klass = self.model.__class__
        init_kwargs = pmodel._get_init_kwds()
        if vtype == 're':
            init_kwargs['exog_re'] = exog_re
        likev = []
        for x in rvalues:
            model = klass(endog, exog, **init_kwargs)
            if vtype == 're':
                cov_re = params.cov_re.copy()
                cov_re[0, 0] = x
                params.cov_re = cov_re
            else:
                params.vcomp[re_ix] = x
            rslt = model.fit(start_params=params, free=free, reml=self.reml, cov_pen=self.cov_pen, **fit_kwargs)._results
            likev.append([x * rslt.scale, rslt.llf])
        likev = np.asarray(likev)
        return likev