from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
class AnovaResults:
    """
    Anova results class

    Attributes
    ----------
    anova_table : DataFrame
    """

    def __init__(self, anova_table):
        self.anova_table = anova_table

    def __str__(self):
        return self.summary().__str__()

    def summary(self):
        """create summary results

        Returns
        -------
        summary : summary2.Summary instance
        """
        summ = summary2.Summary()
        summ.add_title('Anova')
        summ.add_df(self.anova_table)
        return summ