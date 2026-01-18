import numpy as np
from numpy.testing import assert_equal, assert_raises
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.grouputils import GroupSorted
class CheckPanelLagMixin:

    @classmethod
    def calculate(cls):
        cls.g = g = GroupSorted(cls.gind)
        cls.alla = [(lag, sw.lagged_groups(cls.x, lag, g.groupidx)) for lag in range(5)]

    def test_values(self):
        for lag, (y0, ylag) in self.alla:
            assert_equal(y0, self.alle[lag].T)
            assert_equal(y0, ylag + lag)

    def test_raises(self):
        mlag = self.mlag
        assert_raises(ValueError, sw.lagged_groups, self.x, mlag, self.g.groupidx)