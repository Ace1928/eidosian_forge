import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class TestSim1(CheckExternalMixin):
    mean = 0.401499
    sum = 12.9553441
    var = 1.08022
    std = 1.03933
    quantiles = np.r_[-1.81098, -0.84052, 0.32859, 0.77808, 2.93431]

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        cls.data = np.random.normal(size=20)
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()