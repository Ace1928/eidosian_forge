import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class Test2x2_1(Check2x2Mixin):

    @classmethod
    def setup_class(cls):
        data = np.zeros((8, 2))
        data[:, 0] = [0, 0, 1, 1, 0, 0, 1, 1]
        data[:, 1] = [0, 1, 0, 1, 0, 1, 0, 1]
        cls.data = np.asarray(data)
        cls.table = np.asarray([[2, 2], [2, 2]])
        cls.oddsratio = 1.0
        cls.log_oddsratio = 0.0
        cls.log_oddsratio_se = np.sqrt(2)
        cls.oddsratio_confint = [0.06254883616611233, 15.98750770268975]
        cls.oddsratio_pvalue = 1.0
        cls.riskratio = 1.0
        cls.log_riskratio = 0.0
        cls.log_riskratio_se = 1 / np.sqrt(2)
        cls.riskratio_pvalue = 1.0
        cls.riskratio_confint = [0.2500976532599063, 3.9984381579173824]
        cls.log_riskratio_confint = [-1.3859038243496782, 1.3859038243496782]
        ss = ['               Estimate   SE   LCB    UCB   p-value', '---------------------------------------------------', 'Odds ratio        1.000        0.063 15.988   1.000', 'Log odds ratio    0.000 1.414 -2.772  2.772   1.000', 'Risk ratio        1.000        0.250  3.998   1.000', 'Log risk ratio    0.000 0.707 -1.386  1.386   1.000', '---------------------------------------------------']
        cls.summary_string = '\n'.join(ss)
        cls.initialize()