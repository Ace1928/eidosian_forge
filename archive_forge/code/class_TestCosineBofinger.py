import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
class TestCosineBofinger(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('cos', 'bofinger')