import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
class ContrastResults:
    """
    Class for results of tests of linear restrictions on coefficients in a model.

    This class functions mainly as a container for `t_test`, `f_test` and
    `wald_test` for the parameters of a model.

    The attributes depend on the statistical test and are either based on the
    normal, the t, the F or the chisquare distribution.
    """

    def __init__(self, t=None, F=None, sd=None, effect=None, df_denom=None, df_num=None, alpha=0.05, **kwds):
        self.effect = effect
        if F is not None:
            self.distribution = 'F'
            self.fvalue = F
            self.statistic = self.fvalue
            self.df_denom = df_denom
            self.df_num = df_num
            self.dist = fdist
            self.dist_args = (df_num, df_denom)
            self.pvalue = fdist.sf(F, df_num, df_denom)
        elif t is not None:
            self.distribution = 't'
            self.tvalue = t
            self.statistic = t
            self.sd = sd
            self.df_denom = df_denom
            self.dist = student_t
            self.dist_args = (df_denom,)
            self.pvalue = self.dist.sf(np.abs(t), df_denom) * 2
        elif 'statistic' in kwds:
            self.distribution = kwds['distribution']
            self.statistic = kwds['statistic']
            self.tvalue = value = kwds['statistic']
            self.sd = sd
            self.dist = getattr(stats, self.distribution)
            self.dist_args = kwds.get('dist_args', ())
            if self.distribution == 'chi2':
                self.pvalue = self.dist.sf(self.statistic, df_denom)
                self.df_denom = df_denom
            else:
                'normal'
                self.pvalue = np.full_like(value, np.nan)
                not_nan = ~np.isnan(value)
                self.pvalue[not_nan] = self.dist.sf(np.abs(value[not_nan])) * 2
        else:
            self.pvalue = np.nan
        self.pvalue = np.squeeze(self.pvalue)
        if self.effect is not None:
            self.c_names = ['c%d' % ii for ii in range(len(self.effect))]
        else:
            self.c_names = None

    def conf_int(self, alpha=0.05):
        """
        Returns the confidence interval of the value, `effect` of the constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        if self.effect is not None:
            q = self.dist.ppf(1 - alpha / 2.0, *self.dist_args)
            lower = self.effect - q * self.sd
            upper = self.effect + q * self.sd
            return np.column_stack((lower, upper))
        else:
            raise NotImplementedError('Confidence Interval not available')

    def __str__(self):
        return self.summary().__str__()

    def __repr__(self):
        return str(self.__class__) + '\n' + self.__str__()

    def summary(self, xname=None, alpha=0.05, title=None):
        """Summarize the Results of the hypothesis test

        Parameters
        ----------
        xname : list[str], optional
            Default is `c_##` for ## in the number of regressors
        alpha : float
            significance level for the confidence intervals. Default is
            alpha = 0.05 which implies a confidence level of 95%.
        title : str, optional
            Title for the params table. If not None, then this replaces the
            default title

        Returns
        -------
        smry : str or Summary instance
            This contains a parameter results table in the case of t or z test
            in the same form as the parameter results table in the model
            results summary.
            For F or Wald test, the return is a string.
        """
        if self.effect is not None:
            if title is None:
                title = 'Test for Constraints'
            elif title == '':
                title = None
            use_t = self.distribution == 't'
            yname = 'constraints'
            if xname is None:
                xname = self.c_names
            from statsmodels.iolib.summary import summary_params
            pvalues = np.atleast_1d(self.pvalue)
            summ = summary_params((self, self.effect, self.sd, self.statistic, pvalues, self.conf_int(alpha)), yname=yname, xname=xname, use_t=use_t, title=title, alpha=alpha)
            return summ
        elif hasattr(self, 'fvalue'):
            return '<F test: F=%s, p=%s, df_denom=%.3g, df_num=%.3g>' % (repr(self.fvalue), self.pvalue, self.df_denom, self.df_num)
        elif self.distribution == 'chi2':
            return '<Wald test (%s): statistic=%s, p-value=%s, df_denom=%.3g>' % (self.distribution, self.statistic, self.pvalue, self.df_denom)
        else:
            return '<Wald test: statistic=%s, p-value=%s>' % (self.statistic, self.pvalue)

    def summary_frame(self, xname=None, alpha=0.05):
        """Return the parameter table as a pandas DataFrame

        This is only available for t and normal tests
        """
        if self.effect is not None:
            use_t = self.distribution == 't'
            yname = 'constraints'
            if xname is None:
                xname = self.c_names
            from statsmodels.iolib.summary import summary_params_frame
            summ = summary_params_frame((self, self.effect, self.sd, self.statistic, self.pvalue, self.conf_int(alpha)), yname=yname, xname=xname, use_t=use_t, alpha=alpha)
            return summ
        else:
            raise NotImplementedError('only available for t and z')