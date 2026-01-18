from two mean values what can be explained by the data and
from textwrap import dedent
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
class OaxacaResults:
    """
    This class summarizes the fit of the OaxacaBlinder model.

    Use .summary() to get a table of the fitted values or
    use .params to receive a list of the values
    use .std to receive a list of the standard errors

    If a two-fold model was fitted, this will return
    unexplained effect, explained effect, and the
    mean gap. The list will always be of the following order
    and type. If standard error was asked for, then standard error
    calculations will also be included for each variable after each
    calculated effect.

    unexplained : float
        This is the effect that cannot be explained by the data at hand.
        This does not mean it cannot be explained with more.
    explained : float
        This is the effect that can be explained using the data.
    gap : float
        This is the gap in the mean differences of the two groups.

    If a three-fold model was fitted, this will
    return characteristic effect, coefficient effect
    interaction effect, and the mean gap. The list will
    be of the following order and type. If standard error was asked
    for, then standard error calculations will also be included for
    each variable after each calculated effect.

    endowment effect : float
        This is the effect due to the group differences in
        predictors
    coefficient effect : float
        This is the effect due to differences of the coefficients
        of the two groups
    interaction effect : float
        This is the effect due to differences in both effects
        existing at the same time between the two groups.
    gap : float
        This is the gap in the mean differences of the two groups.

    Attributes
    ----------
    params
        A list of all values for the fitted models.
    std
        A list of standard error calculations.
    """

    def __init__(self, results, model_type, std_val=None):
        self.params = results
        self.std = std_val
        self.model_type = model_type

    def summary(self):
        """
        Print a summary table with the Oaxaca-Blinder effects
        """
        if self.model_type == 2:
            if self.std is None:
                print(dedent(f'                Oaxaca-Blinder Two-fold Effects\n                Unexplained Effect: {self.params[0]:.5f}\n                Explained Effect: {self.params[1]:.5f}\n                Gap: {self.params[2]:.5f}'))
            else:
                print(dedent('                Oaxaca-Blinder Two-fold Effects\n                Unexplained Effect: {:.5f}\n                Unexplained Standard Error: {:.5f}\n                Explained Effect: {:.5f}\n                Explained Standard Error: {:.5f}\n                Gap: {:.5f}'.format(self.params[0], self.std[0], self.params[1], self.std[1], self.params[2])))
        if self.model_type == 3:
            if self.std is None:
                print(dedent(f'                Oaxaca-Blinder Three-fold Effects\n                Endowment Effect: {self.params[0]:.5f}\n                Coefficient Effect: {self.params[1]:.5f}\n                Interaction Effect: {self.params[2]:.5f}\n                Gap: {self.params[3]:.5f}'))
            else:
                print(dedent(f'                Oaxaca-Blinder Three-fold Effects\n                Endowment Effect: {self.params[0]:.5f}\n                Endowment Standard Error: {self.std[0]:.5f}\n                Coefficient Effect: {self.params[1]:.5f}\n                Coefficient Standard Error: {self.std[1]:.5f}\n                Interaction Effect: {self.params[2]:.5f}\n                Interaction Standard Error: {self.std[2]:.5f}\n                Gap: {self.params[3]:.5f}'))