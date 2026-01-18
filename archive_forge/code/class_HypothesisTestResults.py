import numpy as np
from statsmodels.iolib.table import SimpleTable
class HypothesisTestResults:
    """
    Results class for hypothesis tests.

    Parameters
    ----------
    test_statistic : float
    crit_value : float
    pvalue : float, 0 <= `pvalue` <= 1
    df : int
        Degrees of freedom.
    signif : float, 0 < `signif` < 1
        Significance level.
    method : str
        The kind of test (e.g. ``"f"`` for F-test, ``"wald"`` for Wald-test).
    title : str
        A title describing the test. It will be part of the summary.
    h0 : str
        A string describing the null hypothesis. It will be used in the
        summary.
    """

    def __init__(self, test_statistic, crit_value, pvalue, df, signif, method, title, h0):
        self.test_statistic = test_statistic
        self.crit_value = crit_value
        self.pvalue = pvalue
        self.df = df
        self.signif = signif
        self.method = method.capitalize()
        if test_statistic < crit_value:
            self.conclusion = 'fail to reject'
        else:
            self.conclusion = 'reject'
        self.title = title
        self.h0 = h0
        self.conclusion_str = 'Conclusion: %s H_0' % self.conclusion
        self.signif_str = f' at {self.signif:.0%} significance level'

    def summary(self):
        """Return summary"""
        title = self.title + '. ' + self.h0 + '. ' + self.conclusion_str + self.signif_str + '.'
        data_fmt = {'data_fmts': ['%#0.4g', '%#0.4g', '%#0.3F', '%s']}
        html_data_fmt = dict(data_fmt)
        html_data_fmt['data_fmts'] = ['<td>' + i + '</td>' for i in html_data_fmt['data_fmts']]
        return SimpleTable(data=[[self.test_statistic, self.crit_value, self.pvalue, str(self.df)]], headers=['Test statistic', 'Critical value', 'p-value', 'df'], title=title, txt_fmt=data_fmt, html_fmt=html_data_fmt, ltx_fmt=data_fmt)

    def __str__(self):
        return '<' + self.__module__ + '.' + self.__class__.__name__ + ' object. ' + self.h0 + ': ' + self.conclusion + self.signif_str + f'. Test statistic: {self.test_statistic:.3f}' + f', critical value: {self.crit_value:.3f}>' + f', p-value: {self.pvalue:.3f}>'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return np.allclose(self.test_statistic, other.test_statistic) and np.allclose(self.crit_value, other.crit_value) and np.allclose(self.pvalue, other.pvalue) and np.allclose(self.signif, other.signif)