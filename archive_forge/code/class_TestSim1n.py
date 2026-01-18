import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class TestSim1n(CheckExternalMixin):
    mean = -0.3131058
    sum = -6.2621168
    var = 0.49722696
    std = 0.70514322
    sem = 0.15767482
    quantiles = np.r_[-1.61593, -1.45576, -0.24356, 0.1677, 1.18791]

    @classmethod
    def setup_class(cls):
        np.random.seed(4342)
        cls.data = np.random.normal(size=20)
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.weights *= 20 / cls.weights.sum()
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives(1)