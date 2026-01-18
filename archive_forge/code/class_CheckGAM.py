from statsmodels.compat.python import lrange
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats
import pytest
from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod.families import family, links
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
class CheckGAM(CheckAM):

    def test_mu(self):
        assert_almost_equal(self.res1.mu_pred, self.res2.mu_pred, decimal=0)

    def test_prediction(self):
        assert_almost_equal(self.res1.y_predshort, self.res2.y_pred[:10], decimal=2)