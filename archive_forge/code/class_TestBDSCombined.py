import os
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from statsmodels.tsa.stattools import bds
class TestBDSCombined(CheckBDS):
    """
    BDS Test on np.r_[np.random.normal(size=25), np.random.uniform(size=25)]
    """

    @classmethod
    def setup_class(cls):
        cls.results = results[results[0] == 3]
        cls.bds_stats = np.array(cls.results[2].iloc[1:])
        cls.pvalues = np.array(cls.results[3].iloc[1:])
        cls.data = data[2][data[2].notnull()]
        cls.res = bds(cls.data, 5)