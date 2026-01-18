import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
class TestSquaredTF(CheckDistEquivalence):

    def __init__(self):
        self.dist = squaretg
        self.trargs = (10,)
        self.trkwds = {}
        self.statsdist = stats.f
        self.stargs = (1, 10)
        self.stkwds = {}