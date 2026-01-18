import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class TestStratified3(CheckStratifiedMixin):
    """
    library(DescTools)
    data = array(c(313, 512, 19, 89,
                   207, 353, 8, 17,
                   205, 120, 391, 202,
                   278, 139, 244, 131,
                   138, 53, 299, 94,
                   351, 22, 317, 24),
                   dim=c(2, 2, 6))
    rslt = mantelhaen.test(data)
    bd1 = BreslowDayTest(data, correct=FALSE)
    bd2 = BreslowDayTest(data, correct=TRUE)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 6
        tables[0] = np.array([[313, 512], [19, 89]])
        tables[1] = np.array([[207, 353], [8, 17]])
        tables[2] = np.array([[205, 120], [391, 202]])
        tables[3] = np.array([[278, 139], [244, 131]])
        tables[4] = np.array([[138, 53], [299, 94]])
        tables[5] = np.array([[351, 22], [317, 24]])
        cls.initialize(tables)
        cls.oddsratio_pooled = 1.101879
        cls.logodds_pooled = np.log(1.101879)
        cls.mh_stat = 1.3368
        cls.mh_pvalue = 0.2476
        cls.or_lcb = 0.9402012
        cls.or_ucb = 1.2913602
        cls.or_homog = 18.83297
        cls.or_homog_p = 0.002064786
        cls.or_homog_adj = 18.83297
        cls.or_homog_adj_p = 0.002064786