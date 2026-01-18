import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.tools import pca
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
class FactorModelUnivariate:
    """

    Todo:
    check treatment of const, make it optional ?
        add hasconst (0 or 1), needed when selecting nfact+hasconst
    options are arguments in calc_factors, should be more public instead
    cross-validation is slow for large number of observations
    """

    def __init__(self, endog, exog):
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)

    def calc_factors(self, x=None, keepdim=0, addconst=True):
        """get factor decomposition of exogenous variables

        This uses principal component analysis to obtain the factors. The number
        of factors kept is the maximum that will be considered in the regression.
        """
        if x is None:
            x = self.exog
        else:
            x = np.asarray(x)
        xred, fact, evals, evecs = pca(x, keepdim=keepdim, normalize=1)
        self.exog_reduced = xred
        if addconst:
            self.factors = sm.add_constant(fact, prepend=True)
            self.hasconst = 1
        else:
            self.factors = fact
            self.hasconst = 0
        self.evals = evals
        self.evecs = evecs

    def fit_fixed_nfact(self, nfact):
        if not hasattr(self, 'factors_wconst'):
            self.calc_factors()
        return sm.OLS(self.endog, self.factors[:, :nfact + 1]).fit()

    def fit_find_nfact(self, maxfact=None, skip_crossval=True, cv_iter=None):
        """estimate the model and selection criteria for up to maxfact factors

        The selection criteria that are calculated are AIC, BIC, and R2_adj. and
        additionally cross-validation prediction error sum of squares if `skip_crossval`
        is false. Cross-validation is not used by default because it can be
        time consuming to calculate.

        By default the cross-validation method is Leave-one-out on the full dataset.
        A different cross-validation sample can be specified as an argument to
        cv_iter.

        Results are attached in `results_find_nfact`



        """
        if not hasattr(self, 'factors'):
            self.calc_factors()
        hasconst = self.hasconst
        if maxfact is None:
            maxfact = self.factors.shape[1] - hasconst
        if maxfact + hasconst < 1:
            raise ValueError('nothing to do, number of factors (incl. constant) should ' + 'be at least 1')
        maxfact = min(maxfact, 10)
        y0 = self.endog
        results = []
        for k in range(1, maxfact + hasconst):
            fact = self.factors[:, :k]
            res = sm.OLS(y0, fact).fit()
            if not skip_crossval:
                if cv_iter is None:
                    cv_iter = LeaveOneOut(len(y0))
                prederr2 = 0.0
                for inidx, outidx in cv_iter:
                    res_l1o = sm.OLS(y0[inidx], fact[inidx, :]).fit()
                    prederr2 += (y0[outidx] - res_l1o.model.predict(res_l1o.params, fact[outidx, :])) ** 2.0
            else:
                prederr2 = np.nan
            results.append([k, res.aic, res.bic, res.rsquared_adj, prederr2])
        self.results_find_nfact = results = np.array(results)
        self.best_nfact = np.r_[np.argmin(results[:, 1:3], 0), np.argmax(results[:, 3], 0), np.argmin(results[:, -1], 0)]

    def summary_find_nfact(self):
        """provides a summary for the selection of the number of factors

        Returns
        -------
        sumstr : str
            summary of the results for selecting the number of factors

        """
        if not hasattr(self, 'results_find_nfact'):
            self.fit_find_nfact()
        results = self.results_find_nfact
        sumstr = ''
        sumstr += '\n' + 'Best result for k, by AIC, BIC, R2_adj, L1O'
        sumstr += '\n' + ' ' * 19 + '%5d %4d %6d %5d' % tuple(self.best_nfact)
        from statsmodels.iolib.table import SimpleTable
        headers = 'k, AIC, BIC, R2_adj, L1O'.split(', ')
        numformat = ['%6d'] + ['%10.3f'] * 4
        txt_fmt1 = dict(data_fmts=numformat)
        tabl = SimpleTable(results, headers, None, txt_fmt=txt_fmt1)
        sumstr += '\n' + 'PCA regression on simulated data,'
        sumstr += '\n' + 'DGP: 2 factors and 4 explanatory variables'
        sumstr += '\n' + tabl.__str__()
        sumstr += '\n' + 'Notes: k is number of components of PCA,'
        sumstr += '\n' + '       constant is added additionally'
        sumstr += '\n' + '       k=0 means regression on constant only'
        sumstr += '\n' + '       L1O: sum of squared prediction errors for leave-one-out'
        return sumstr