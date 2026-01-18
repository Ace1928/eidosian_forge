import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestDiscretizedBurr12(CheckDiscretized):

    @classmethod
    def setup_class(cls):
        cls.d_offset = 0
        cls.ddistr = stats.burr12
        cls.paramg = (2, 1, 0, 1.5)
        cls.paramd = (2, 1, 1.5)
        cls.shapes = 'c, d, s'
        cls.start_params = (0.5, 1, 0.5)