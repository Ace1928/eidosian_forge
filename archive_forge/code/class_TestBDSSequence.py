import os
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from statsmodels.tsa.stattools import bds
class TestBDSSequence(CheckBDS):
    """
    BDS Test on np.arange(1,26)
    """

    @classmethod
    def setup_class(cls):
        cls.results = results[results[0] == 1]
        cls.bds_stats = np.array(cls.results[2].iloc[1:])
        cls.pvalues = np.array(cls.results[3].iloc[1:])
        cls.data = data[0][data[0].notnull()]
        cls.res = bds(cls.data, 5)