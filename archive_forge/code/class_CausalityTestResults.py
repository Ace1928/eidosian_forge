import numpy as np
from statsmodels.iolib.table import SimpleTable
class CausalityTestResults(HypothesisTestResults):
    """
    Results class for Granger-causality and instantaneous causality.

    Parameters
    ----------
    causing : list of str
        This list contains the potentially causing variables.
    caused : list of str
        This list contains the potentially caused variables.
    test_statistic : float
    crit_value : float
    pvalue : float
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    test : str {``"granger"``, ``"inst"``}, default: ``"granger"``
        If ``"granger"``, Granger-causality has been tested. If ``"inst"``,
        instantaneous causality has been tested.
    method : str {``"f"``, ``"wald"``}
        The kind of test. ``"f"`` indicates an F-test, ``"wald"`` indicates a
        Wald-test.
    """

    def __init__(self, causing, caused, test_statistic, crit_value, pvalue, df, signif, test='granger', method=None):
        self.causing = causing
        self.caused = caused
        self.test = test
        if method is None or method.lower() not in ['f', 'wald']:
            raise ValueError('The method ("f" for F-test, "wald" for Wald-test) must not be None.')
        method = method.capitalize()
        title = 'Granger' if self.test == 'granger' else 'Instantaneous'
        title += ' causality %s-test' % method
        h0 = 'H_0: '
        if len(self.causing) == 1:
            h0 += f'{self.causing[0]} does not '
        else:
            h0 += f'{self.causing} do not '
        h0 += 'Granger-' if self.test == 'granger' else 'instantaneously '
        h0 += 'cause '
        if len(self.caused) == 1:
            h0 += self.caused[0]
        else:
            h0 += '[' + ', '.join(caused) + ']'
        super().__init__(test_statistic, crit_value, pvalue, df, signif, method, title, h0)

    def __eq__(self, other):
        basic_test = super().__eq__(other)
        if not basic_test:
            return False
        test = self.test == other.test
        variables = self.causing == other.causing and self.caused == other.caused
        if not variables and self.test == 'inst':
            variables = self.causing == other.caused and self.caused == other.causing
        return test and variables