import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
class TestSquaredNormChi2_2(CheckDistEquivalence):

    def __init__(self):
        self.dist = squarenormalg
        self.trargs = ()
        self.trkwds = dict(loc=-10, scale=20)
        self.statsdist = stats.chi2
        self.stargs = (1,)
        self.stkwds = dict(loc=-10, scale=20)