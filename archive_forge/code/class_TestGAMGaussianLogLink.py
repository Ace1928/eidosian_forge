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
@pytest.mark.xfail(reason='Unknown, results do not match expected.', strict=True, raises=AssertionError)
class TestGAMGaussianLogLink(BaseGAM):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.family = family.Gaussian(links.Log())
        cls.rvs = stats.norm.rvs
        cls.scale = 5
        cls.init()