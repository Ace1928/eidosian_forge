import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
class TestGradLogit(CheckGradLoglikeMixin):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.mod = sm.Logit(data.endog, data.exog)
        cls.params = [np.array([1, 0.25, 1.4, -7])]