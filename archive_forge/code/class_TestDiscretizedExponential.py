import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestDiscretizedExponential(CheckDiscretized):

    @classmethod
    def setup_class(cls):
        cls.d_offset = 0
        cls.ddistr = stats.expon
        cls.paramg = (0, 5)
        cls.paramd = (5,)
        cls.shapes = 's'
        cls.start_params = 0.5