import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class TestStratified2(CheckStratifiedMixin):
    """
    library(DescTools)
    data = array(c(20, 14, 10, 24,
                   15, 12, 3, 15,
                   3, 2, 3, 2,
                   12, 3, 7, 5,
                   1, 0, 3, 2),
                   dim=c(2, 2, 5))
    rslt = mantelhaen.test(data)
    bd1 = BreslowDayTest(data, correct=FALSE)
    bd2 = BreslowDayTest(data, correct=TRUE)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 5
        tables[0] = np.array([[20, 14], [10, 24]])
        tables[1] = np.array([[15, 12], [3, 15]])
        tables[2] = np.array([[3, 2], [3, 2]])
        tables[3] = np.array([[12, 3], [7, 5]])
        tables[4] = np.array([[1, 0], [3, 2]])
        cls.initialize(tables, use_arr=True)
        cls.oddsratio_pooled = 3.5912
        cls.logodds_pooled = np.log(3.5912)
        cls.mh_stat = 11.8852
        cls.mh_pvalue = 0.0005658
        cls.or_lcb = 1.781135
        cls.or_ucb = 7.240633
        cls.or_homog = 1.8438
        cls.or_homog_p = 0.7645
        cls.or_homog_adj = 1.8436
        cls.or_homog_adj_p = 0.7645