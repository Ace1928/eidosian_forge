import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
class CompareMeans:
    """class for two sample comparison

    The tests and the confidence interval work for multi-endpoint comparison:
    If d1 and d2 have the same number of rows, then each column of the data
    in d1 is compared with the corresponding column in d2.

    Parameters
    ----------
    d1, d2 : instances of DescrStatsW

    Notes
    -----
    The result for the statistical tests and the confidence interval are
    independent of the user specified ddof.

    TODO: Extend to any number of groups or write a version that works in that
    case, like in SAS and SPSS.

    """

    def __init__(self, d1, d2):
        """assume d1, d2 hold the relevant attributes

        """
        self.d1 = d1
        self.d2 = d2

    @classmethod
    def from_data(cls, data1, data2, weights1=None, weights2=None, ddof1=0, ddof2=0):
        """construct a CompareMeans object from data

        Parameters
        ----------
        data1, data2 : array_like, 1-D or 2-D
            compared datasets
        weights1, weights2 : None or 1-D ndarray
            weights for each observation of data1 and data2 respectively,
            with same length as zero axis of corresponding dataset.
        ddof1, ddof2 : int
            default ddof1=0, ddof2=0, degrees of freedom for data1,
            data2 respectively.

        Returns
        -------
        A CompareMeans instance.

        """
        return cls(DescrStatsW(data1, weights=weights1, ddof=ddof1), DescrStatsW(data2, weights=weights2, ddof=ddof2))

    def summary(self, use_t=True, alpha=0.05, usevar='pooled', value=0):
        """summarize the results of the hypothesis test

        Parameters
        ----------
        use_t : bool, optional
            if use_t is True, then t test results are returned
            if use_t is False, then z test results are returned
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is
            assumed to be the same. If ``unequal``, then the variance of
            Welch ttest will be used, and the degrees of freedom are those
            of Satterthwaite if ``use_t`` is True.
        value : float
            difference between the means under the Null hypothesis.

        Returns
        -------
        smry : SimpleTable

        """
        d1 = self.d1
        d2 = self.d2
        confint_percents = 100 - alpha * 100
        if use_t:
            tstat, pvalue, _ = self.ttest_ind(usevar=usevar, value=value)
            lower, upper = self.tconfint_diff(alpha=alpha, usevar=usevar)
        else:
            tstat, pvalue = self.ztest_ind(usevar=usevar, value=value)
            lower, upper = self.zconfint_diff(alpha=alpha, usevar=usevar)
        if usevar == 'pooled':
            std_err = self.std_meandiff_pooledvar
        else:
            std_err = self.std_meandiff_separatevar
        std_err = np.atleast_1d(std_err)
        tstat = np.atleast_1d(tstat)
        pvalue = np.atleast_1d(pvalue)
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        conf_int = np.column_stack((lower, upper))
        params = np.atleast_1d(d1.mean - d2.mean - value)
        title = 'Test for equality of means'
        yname = 'y'
        xname = ['subset #%d' % (ii + 1) for ii in range(tstat.shape[0])]
        from statsmodels.iolib.summary import summary_params
        return summary_params((None, params, std_err, tstat, pvalue, conf_int), alpha=alpha, use_t=use_t, yname=yname, xname=xname, title=title)

    @cache_readonly
    def std_meandiff_separatevar(self):
        d1 = self.d1
        d2 = self.d2
        return np.sqrt(d1._var / (d1.nobs - 1) + d2._var / (d2.nobs - 1))

    @cache_readonly
    def std_meandiff_pooledvar(self):
        """variance assuming equal variance in both data sets

        """
        d1 = self.d1
        d2 = self.d2
        var_pooled = (d1.sumsquares + d2.sumsquares) / (d1.nobs - 1 + d2.nobs - 1)
        return np.sqrt(var_pooled * (1.0 / d1.nobs + 1.0 / d2.nobs))

    def dof_satt(self):
        """degrees of freedom of Satterthwaite for unequal variance
        """
        d1 = self.d1
        d2 = self.d2
        sem1 = d1._var / (d1.nobs - 1)
        sem2 = d2._var / (d2.nobs - 1)
        semsum = sem1 + sem2
        z1 = (sem1 / semsum) ** 2 / (d1.nobs - 1)
        z2 = (sem2 / semsum) ** 2 / (d2.nobs - 1)
        dof = 1.0 / (z1 + z2)
        return dof

    def ttest_ind(self, alternative='two-sided', usevar='pooled', value=0):
        """ttest for the null hypothesis of identical means

        this should also be the same as onewaygls, except for ddof differences

        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples, see notes for 2-D case
        x2 : array_like, 1-D or 2-D
            second of the two independent samples, see notes for 2-D case
        alternative : str
            The alternative hypothesis, H1, has to be one of the following
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used
        value : float
            difference between the means under the Null hypothesis.


        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test
        df : int or float
            degrees of freedom used in the t-test

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        d1 = self.d1
        d2 = self.d2
        if usevar == 'pooled':
            stdm = self.std_meandiff_pooledvar
            dof = d1.nobs - 1 + d2.nobs - 1
        elif usevar == 'unequal':
            stdm = self.std_meandiff_separatevar
            dof = self.dof_satt()
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        tstat, pval = _tstat_generic(d1.mean, d2.mean, stdm, dof, alternative, diff=value)
        return (tstat, pval, dof)

    def ztest_ind(self, alternative='two-sided', usevar='pooled', value=0):
        """z-test for the null hypothesis of identical means

        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples, see notes for 2-D case
        x2 : array_like, 1-D or 2-D
            second of the two independent samples, see notes for 2-D case
        alternative : str
            The alternative hypothesis, H1, has to be one of the following
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then the standard deviations of the samples may
            be different.
        value : float
            difference between the means under the Null hypothesis.

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the z-test

        """
        d1 = self.d1
        d2 = self.d2
        if usevar == 'pooled':
            stdm = self.std_meandiff_pooledvar
        elif usevar == 'unequal':
            stdm = self.std_meandiff_separatevar
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        tstat, pval = _zstat_generic(d1.mean, d2.mean, stdm, alternative, diff=value)
        return (tstat, pval)

    def tconfint_diff(self, alpha=0.05, alternative='two-sided', usevar='pooled'):
        """confidence interval for the difference in means

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :

            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        lower, upper : floats
            lower and upper limits of the confidence interval

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        d1 = self.d1
        d2 = self.d2
        diff = d1.mean - d2.mean
        if usevar == 'pooled':
            std_diff = self.std_meandiff_pooledvar
            dof = d1.nobs - 1 + d2.nobs - 1
        elif usevar == 'unequal':
            std_diff = self.std_meandiff_separatevar
            dof = self.dof_satt()
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        res = _tconfint_generic(diff, std_diff, dof, alpha=alpha, alternative=alternative)
        return res

    def zconfint_diff(self, alpha=0.05, alternative='two-sided', usevar='pooled'):
        """confidence interval for the difference in means

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :

            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        lower, upper : floats
            lower and upper limits of the confidence interval

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        d1 = self.d1
        d2 = self.d2
        diff = d1.mean - d2.mean
        if usevar == 'pooled':
            std_diff = self.std_meandiff_pooledvar
        elif usevar == 'unequal':
            std_diff = self.std_meandiff_separatevar
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        res = _zconfint_generic(diff, std_diff, alpha=alpha, alternative=alternative)
        return res

    def ttost_ind(self, low, upp, usevar='pooled'):
        """
        test of equivalence for two independent samples, base on t-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        """
        tt1 = self.ttest_ind(alternative='larger', usevar=usevar, value=low)
        tt2 = self.ttest_ind(alternative='smaller', usevar=usevar, value=upp)
        return (np.maximum(tt1[1], tt2[1]), (tt1, tt2))

    def ztost_ind(self, low, upp, usevar='pooled'):
        """
        test of equivalence for two independent samples, based on z-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        """
        tt1 = self.ztest_ind(alternative='larger', usevar=usevar, value=low)
        tt2 = self.ztest_ind(alternative='smaller', usevar=usevar, value=upp)
        return (np.maximum(tt1[1], tt2[1]), tt1, tt2)