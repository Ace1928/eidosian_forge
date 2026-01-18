import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
class RankCompareResult(HolderTuple):
    """Results for rank comparison

    This is a subclass of HolderTuple that includes results from intermediate
    computations, as well as methods for hypothesis tests, confidence intervals
    and summary.
    """

    def conf_int(self, value=None, alpha=0.05, alternative='two-sided'):
        """
        Confidence interval for probability that sample 1 has larger values

        Confidence interval is for the shifted probability

            P(x1 > x2) + 0.5 * P(x1 = x2) - value

        Parameters
        ----------
        value : float
            Value, default 0, shifts the confidence interval,
            e.g. ``value=0.5`` centers the confidence interval at zero.
        alpha : float
            Significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        lower : float or ndarray
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        upper : float or ndarray
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".

        """
        p0 = value
        if p0 is None:
            p0 = 0
        diff = self.prob1 - p0
        std_diff = np.sqrt(self.var / self.nobs)
        if self.use_t is False:
            return _zconfint_generic(diff, std_diff, alpha, alternative)
        else:
            return _tconfint_generic(diff, std_diff, self.df, alpha, alternative)

    def test_prob_superior(self, value=0.5, alternative='two-sided'):
        """test for superiority probability

        H0: P(x1 > x2) + 0.5 * P(x1 = x2) = value

        The alternative is that the probability is either not equal, larger
        or smaller than the null-value depending on the chosen alternative.

        Parameters
        ----------
        value : float
            Value of the probability under the Null hypothesis.
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        res : HolderTuple
            HolderTuple instance with the following main attributes

            statistic : float
                Test statistic for z- or t-test
            pvalue : float
                Pvalue of the test based on either normal or t distribution.

        """
        p0 = value
        std_diff = np.sqrt(self.var / self.nobs)
        if not self.use_t:
            stat, pv = _zstat_generic(self.prob1, p0, std_diff, alternative, diff=0)
            distr = 'normal'
        else:
            stat, pv = _tstat_generic(self.prob1, p0, std_diff, self.df, alternative, diff=0)
            distr = 't'
        res = HolderTuple(statistic=stat, pvalue=pv, df=self.df, distribution=distr)
        return res

    def tost_prob_superior(self, low, upp):
        """test of stochastic (non-)equivalence of p = P(x1 > x2)

        Null hypothesis:  p < low or p > upp
        Alternative hypothesis:  low < p < upp

        where p is the probability that a random draw from the population of
        the first sample has a larger value than a random draw from the
        population of the second sample, specifically

            p = P(x1 > x2) + 0.5 * P(x1 = x2)

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the probability p that distribution 1 is stochastically
        superior to distribution 2 is outside of the interval given by
        thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        res : HolderTuple
            HolderTuple instance with the following main attributes

            pvalue : float
                Pvalue of the equivalence test given by the larger pvalue of
                the two one-sided tests.
            statistic : float
                Test statistic of the one-sided test that has the larger
                pvalue.
            results_larger : HolderTuple
                Results instanc with test statistic, pvalue and degrees of
                freedom for lower threshold test.
            results_smaller : HolderTuple
                Results instanc with test statistic, pvalue and degrees of
                freedom for upper threshold test.

        """
        t1 = self.test_prob_superior(low, alternative='larger')
        t2 = self.test_prob_superior(upp, alternative='smaller')
        idx_max = np.asarray(t1.pvalue < t2.pvalue, int)
        title = 'Equivalence test for Prob(x1 > x2) + 0.5 Prob(x1 = x2) '
        res = HolderTuple(statistic=np.choose(idx_max, [t1.statistic, t2.statistic]), pvalue=np.choose(idx_max, [t1.pvalue, t2.pvalue]), results_larger=t1, results_smaller=t2, title=title)
        return res

    def confint_lintransf(self, const=-1, slope=2, alpha=0.05, alternative='two-sided'):
        """confidence interval of a linear transformation of prob1

        This computes the confidence interval for

            d = const + slope * prob1

        Default values correspond to Somers' d.

        Parameters
        ----------
        const, slope : float
            Constant and slope for linear (affine) transformation.
        alpha : float
            Significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        lower : float or ndarray
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        upper : float or ndarray
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".

        """
        low_p, upp_p = self.conf_int(alpha=alpha, alternative=alternative)
        low = const + slope * low_p
        upp = const + slope * upp_p
        if slope < 0:
            low, upp = (upp, low)
        return (low, upp)

    def effectsize_normal(self, prob=None):
        """
        Cohen's d, standardized mean difference under normality assumption.

        This computes the standardized mean difference, Cohen's d, effect size
        that is equivalent to the rank based probability ``p`` of being
        stochastically larger if we assume that the data is normally
        distributed, given by

            :math: `d = F^{-1}(p) * \\sqrt{2}`

        where :math:`F^{-1}` is the inverse of the cdf of the normal
        distribution.

        Parameters
        ----------
        prob : float in (0, 1)
            Probability to be converted to Cohen's d effect size.
            If prob is None, then the ``prob1`` attribute is used.

        Returns
        -------
        equivalent Cohen's d effect size under normality assumption.

        """
        if prob is None:
            prob = self.prob1
        return stats.norm.ppf(prob) * np.sqrt(2)

    def summary(self, alpha=0.05, xname=None):
        """summary table for probability that random draw x1 is larger than x2

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals. Coverage is 1 - alpha
        xname : None or list of str
            If None, then each row has a name column with generic names.
            If xname is a list of strings, then it will be included as part
            of those names.

        Returns
        -------
        SimpleTable instance with methods to convert to different output
        formats.
        """
        yname = 'None'
        effect = np.atleast_1d(self.prob1)
        if self.pvalue is None:
            statistic, pvalue = self.test_prob_superior()
        else:
            pvalue = self.pvalue
            statistic = self.statistic
        pvalues = np.atleast_1d(pvalue)
        ci = np.atleast_2d(self.conf_int(alpha=alpha))
        if ci.shape[0] > 1:
            ci = ci.T
        use_t = self.use_t
        sd = np.atleast_1d(np.sqrt(self.var_prob))
        statistic = np.atleast_1d(statistic)
        if xname is None:
            xname = ['c%d' % ii for ii in range(len(effect))]
        xname2 = ['prob(x1>x2) %s' % ii for ii in xname]
        title = 'Probability sample 1 is stochastically larger'
        from statsmodels.iolib.summary import summary_params
        summ = summary_params((self, effect, sd, statistic, pvalues, ci), yname=yname, xname=xname2, use_t=use_t, title=title, alpha=alpha)
        return summ