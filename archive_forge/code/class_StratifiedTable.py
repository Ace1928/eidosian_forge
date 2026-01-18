import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
class StratifiedTable:
    """
    Analyses for a collection of 2x2 contingency tables.

    Such a collection may arise by stratifying a single 2x2 table with
    respect to another factor.  This class implements the
    'Cochran-Mantel-Haenszel' and 'Breslow-Day' procedures for
    analyzing collections of 2x2 contingency tables.

    Parameters
    ----------
    tables : list or ndarray
        Either a list containing several 2x2 contingency tables, or
        a 2x2xk ndarray in which each slice along the third axis is a
        2x2 contingency table.

    Notes
    -----
    This results are based on a sampling model in which the units are
    independent both within and between strata.
    """

    def __init__(self, tables, shift_zeros=False):
        if isinstance(tables, np.ndarray):
            sp = tables.shape
            if len(sp) != 3 or sp[0] != 2 or sp[1] != 2:
                raise ValueError('If an ndarray, argument must be 2x2xn')
            table = tables * 1.0
        else:
            if any([np.asarray(x).shape != (2, 2) for x in tables]):
                m = 'If `tables` is a list, all of its elements should be 2x2'
                raise ValueError(m)
            table = np.dstack(tables).astype(np.float64)
        if shift_zeros:
            zx = (table == 0).sum(0).sum(0)
            ix = np.flatnonzero(zx > 0)
            if len(ix) > 0:
                table = table.copy()
                table[:, :, ix] += 0.5
        self.table = table
        self._cache = {}
        self._apb = table[0, 0, :] + table[0, 1, :]
        self._apc = table[0, 0, :] + table[1, 0, :]
        self._bpd = table[0, 1, :] + table[1, 1, :]
        self._cpd = table[1, 0, :] + table[1, 1, :]
        self._ad = table[0, 0, :] * table[1, 1, :]
        self._bc = table[0, 1, :] * table[1, 0, :]
        self._apd = table[0, 0, :] + table[1, 1, :]
        self._dma = table[1, 1, :] - table[0, 0, :]
        self._n = table.sum(0).sum(0)

    @classmethod
    def from_data(cls, var1, var2, strata, data):
        """
        Construct a StratifiedTable object from data.

        Parameters
        ----------
        var1 : int or string
            The column index or name of `data` specifying the variable
            defining the rows of the contingency table.  The variable
            must have only two distinct values.
        var2 : int or string
            The column index or name of `data` specifying the variable
            defining the columns of the contingency table.  The variable
            must have only two distinct values.
        strata : int or string
            The column index or name of `data` specifying the variable
            defining the strata.
        data : array_like
            The raw data.  A cross-table for analysis is constructed
            from the first two columns.

        Returns
        -------
        StratifiedTable
        """
        if not isinstance(data, pd.DataFrame):
            data1 = pd.DataFrame(index=np.arange(data.shape[0]), columns=[var1, var2, strata])
            data1[data1.columns[var1]] = data[:, var1]
            data1[data1.columns[var2]] = data[:, var2]
            data1[data1.columns[strata]] = data[:, strata]
        else:
            data1 = data[[var1, var2, strata]]
        gb = data1.groupby(strata).groups
        tables = []
        for g in gb:
            ii = gb[g]
            tab = pd.crosstab(data1.loc[ii, var1], data1.loc[ii, var2])
            if (tab.shape != np.r_[2, 2]).any():
                msg = 'Invalid table dimensions'
                raise ValueError(msg)
            tables.append(np.asarray(tab))
        return cls(tables)

    def test_null_odds(self, correction=False):
        """
        Test that all tables have odds ratio equal to 1.

        This is the 'Mantel-Haenszel' test.

        Parameters
        ----------
        correction : bool
            If True, use the continuity correction when calculating the
            test statistic.

        Returns
        -------
        Bunch
            A bunch containing the chi^2 test statistic and p-value.
        """
        statistic = np.sum(self.table[0, 0, :] - self._apb * self._apc / self._n)
        statistic = np.abs(statistic)
        if correction:
            statistic -= 0.5
        statistic = statistic ** 2
        denom = self._apb * self._apc * self._bpd * self._cpd
        denom /= self._n ** 2 * (self._n - 1)
        denom = np.sum(denom)
        statistic /= denom
        pvalue = 1 - stats.chi2.cdf(statistic, 1)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        return b

    @cache_readonly
    def oddsratio_pooled(self):
        """
        The pooled odds ratio.

        The value is an estimate of a common odds ratio across all of the
        stratified tables.
        """
        odds_ratio = np.sum(self._ad / self._n) / np.sum(self._bc / self._n)
        return odds_ratio

    @cache_readonly
    def logodds_pooled(self):
        """
        Returns the logarithm of the pooled odds ratio.

        See oddsratio_pooled for more information.
        """
        return np.log(self.oddsratio_pooled)

    @cache_readonly
    def riskratio_pooled(self):
        """
        Estimate of the pooled risk ratio.
        """
        acd = self.table[0, 0, :] * self._cpd
        cab = self.table[1, 0, :] * self._apb
        rr = np.sum(acd / self._n) / np.sum(cab / self._n)
        return rr

    @cache_readonly
    def logodds_pooled_se(self):
        """
        Estimated standard error of the pooled log odds ratio

        References
        ----------
        J. Robins, N. Breslow, S. Greenland. "Estimators of the
        Mantel-Haenszel Variance Consistent in Both Sparse Data and
        Large-Strata Limiting Models." Biometrics 42, no. 2 (1986): 311-23.
        """
        adns = np.sum(self._ad / self._n)
        bcns = np.sum(self._bc / self._n)
        lor_va = np.sum(self._apd * self._ad / self._n ** 2) / adns ** 2
        mid = self._apd * self._bc / self._n ** 2
        mid += (1 - self._apd / self._n) * self._ad / self._n
        mid = np.sum(mid)
        mid /= adns * bcns
        lor_va += mid
        lor_va += np.sum((1 - self._apd / self._n) * self._bc / self._n) / bcns ** 2
        lor_va /= 2
        lor_se = np.sqrt(lor_va)
        return lor_se

    def logodds_pooled_confint(self, alpha=0.05, method='normal'):
        """
        A confidence interval for the pooled log odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.

        Returns
        -------
        lcb : float
            The lower confidence limit.
        ucb : float
            The upper confidence limit.
        """
        lor = np.log(self.oddsratio_pooled)
        lor_se = self.logodds_pooled_se
        f = -stats.norm.ppf(alpha / 2)
        lcb = lor - f * lor_se
        ucb = lor + f * lor_se
        return (lcb, ucb)

    def oddsratio_pooled_confint(self, alpha=0.05, method='normal'):
        """
        A confidence interval for the pooled odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.

        Returns
        -------
        lcb : float
            The lower confidence limit.
        ucb : float
            The upper confidence limit.
        """
        lcb, ucb = self.logodds_pooled_confint(alpha, method=method)
        lcb = np.exp(lcb)
        ucb = np.exp(ucb)
        return (lcb, ucb)

    def test_equal_odds(self, adjust=False):
        """
        Test that all odds ratios are identical.

        This is the 'Breslow-Day' testing procedure.

        Parameters
        ----------
        adjust : bool
            Use the 'Tarone' adjustment to achieve the chi^2
            asymptotic distribution.

        Returns
        -------
        A bunch containing the following attributes:

        statistic : float
            The chi^2 test statistic.
        p-value : float
            The p-value for the test.
        """
        table = self.table
        r = self.oddsratio_pooled
        a = 1 - r
        b = r * (self._apb + self._apc) + self._dma
        c = -r * self._apb * self._apc
        dr = np.sqrt(b ** 2 - 4 * a * c)
        e11 = (-b + dr) / (2 * a)
        v11 = 1 / e11 + 1 / (self._apc - e11) + 1 / (self._apb - e11) + 1 / (self._dma + e11)
        v11 = 1 / v11
        statistic = np.sum((table[0, 0, :] - e11) ** 2 / v11)
        if adjust:
            adj = table[0, 0, :].sum() - e11.sum()
            adj = adj ** 2
            adj /= np.sum(v11)
            statistic -= adj
        pvalue = 1 - stats.chi2.cdf(statistic, table.shape[2] - 1)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        return b

    def summary(self, alpha=0.05, float_format='%.3f', method='normal'):
        """
        A summary of all the main results.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence intervals.
        float_format : str
            Used for formatting numeric values in the summary.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """

        def fmt(x):
            if isinstance(x, str):
                return x
            return float_format % x
        co_lcb, co_ucb = self.oddsratio_pooled_confint(alpha=alpha, method=method)
        clo_lcb, clo_ucb = self.logodds_pooled_confint(alpha=alpha, method=method)
        headers = ['Estimate', 'LCB', 'UCB']
        stubs = ['Pooled odds', 'Pooled log odds', 'Pooled risk ratio', '']
        data = [[fmt(x) for x in [self.oddsratio_pooled, co_lcb, co_ucb]], [fmt(x) for x in [self.logodds_pooled, clo_lcb, clo_ucb]], [fmt(x) for x in [self.riskratio_pooled, '', '']], ['', '', '']]
        tab1 = iolib.SimpleTable(data, headers, stubs, data_aligns='r', table_dec_above='')
        headers = ['Statistic', 'P-value', '']
        stubs = ['Test of OR=1', 'Test constant OR']
        rslt1 = self.test_null_odds()
        rslt2 = self.test_equal_odds()
        data = [[fmt(x) for x in [rslt1.statistic, rslt1.pvalue, '']], [fmt(x) for x in [rslt2.statistic, rslt2.pvalue, '']]]
        tab2 = iolib.SimpleTable(data, headers, stubs, data_aligns='r')
        tab1.extend(tab2)
        headers = ['', '', '']
        stubs = ['Number of tables', 'Min n', 'Max n', 'Avg n', 'Total n']
        ss = self.table.sum(0).sum(0)
        data = [['%d' % self.table.shape[2], '', ''], ['%d' % min(ss), '', ''], ['%d' % max(ss), '', ''], ['%.0f' % np.mean(ss), '', ''], ['%d' % sum(ss), '', '', '']]
        tab3 = iolib.SimpleTable(data, headers, stubs, data_aligns='r')
        tab1.extend(tab3)
        return tab1