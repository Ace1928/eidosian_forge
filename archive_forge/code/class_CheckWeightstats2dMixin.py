import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class CheckWeightstats2dMixin(CheckWeightstats1dMixin):

    def test_corr(self):
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(np.corrcoef(x1r.T), d1w.corrcoef, 14)