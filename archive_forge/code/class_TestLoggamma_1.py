import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
class TestLoggamma_1(CheckDistEquivalence):

    def __init__(self):
        self.dist = LogTransf_gen(stats.gamma)
        self.trargs = (10,)
        self.trkwds = {}
        self.statsdist = stats.loggamma
        self.stargs = (10,)
        self.stkwds = {}