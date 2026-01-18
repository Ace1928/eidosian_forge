import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestDiscretizedLomax(CheckDiscretized):

    @classmethod
    def setup_class(cls):
        cls.d_offset = 0
        cls.ddistr = stats.lomax
        cls.paramg = (2, 0, 1.5)
        cls.paramd = (2, 1.5)
        cls.shapes = 'c, s'
        cls.start_params = (0.5, 0.5)