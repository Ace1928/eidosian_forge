from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class IVRegressionResults(RegressionResults):
    """
    Results class for for an OLS model.

    Most of the methods and attributes are inherited from RegressionResults.
    The special methods that are only available for OLS are:

    - get_influence
    - outlier_test
    - el_test
    - conf_int_el

    See Also
    --------
    RegressionResults
    """

    @cache_readonly
    def fvalue(self):
        const_idx = self.model.data.const_idx
        if const_idx is None:
            return np.nan
        else:
            k_vars = len(self.params)
            restriction = np.eye(k_vars)
            idx_noconstant = lrange(k_vars)
            del idx_noconstant[const_idx]
            fval = self.f_test(restriction[idx_noconstant]).fvalue
            return fval

    def spec_hausman(self, dof=None):
        """Hausman's specification test

        See Also
        --------
        spec_hausman : generic function for Hausman's specification test

        """
        endog, exog = (self.model.endog, self.model.exog)
        resols = OLS(endog, exog).fit()
        normalized_cov_params_ols = resols.model.normalized_cov_params
        se2 = resols.ssr / len(endog)
        params_diff = self.params - resols.params
        cov_diff = np.linalg.pinv(self.model.xhatprod) - normalized_cov_params_ols
        if not dof:
            dof = np.linalg.matrix_rank(cov_diff)
        cov_diffpinv = np.linalg.pinv(cov_diff)
        H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff)) / se2
        pval = stats.chi2.sf(H, dof)
        return (H, pval, dof)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Default is `var_##` for ## in p the number of regressors
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """
        from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson
        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)
        wexog = self.model.wexog
        eigvals = np.linalg.eigvalsh(np.dot(wexog.T, wexog))
        eigvals = np.sort(eigvals)
        condno = np.sqrt(eigvals[-1] / eigvals[0])
        self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis, omni=omni, omnipv=omnipv, condno=condno, mineigval=eigvals[0])
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['Two Stage']), ('', ['Least Squares']), ('Date:', None), ('Time:', None), ('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None)]
        top_right = [('R-squared:', ['%#8.3f' % self.rsquared]), ('Adj. R-squared:', ['%#8.3f' % self.rsquared_adj]), ('F-statistic:', ['%#8.4g' % self.fvalue]), ('Prob (F-statistic):', ['%#6.3g' % self.f_pvalue])]
        diagn_left = [('Omnibus:', ['%#6.3f' % omni]), ('Prob(Omnibus):', ['%#6.3f' % omnipv]), ('Skew:', ['%#6.3f' % skew]), ('Kurtosis:', ['%#6.3f' % kurtosis])]
        diagn_right = [('Durbin-Watson:', ['%#8.3f' % durbin_watson(self.wresid)]), ('Jarque-Bera (JB):', ['%#8.3f' % jb]), ('Prob(JB):', ['%#8.3g' % jbpv]), ('Cond. No.', ['%#8.3g' % condno])]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=True)
        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right, yname=yname, xname=xname, title='')
        return smry