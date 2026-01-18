import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
class TestEpanechnikovHsheatherQ75(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.engel.load_pandas().data
        y, X = dmatrices('foodexp ~ income', data, return_type='dataframe')
        cls.res1 = QuantReg(y, X).fit(q=0.75, vcov='iid', kernel='epa', bandwidth='hsheather')
        cls.res2 = epanechnikov_hsheather_q75