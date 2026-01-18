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
@pytest.mark.xfail(reason='Passing wrong number of args/kwargs to _parse_args_rvs', strict=True, raises=TypeError)
class TestGAMNegativeBinomial(BaseGAM):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.family = family.NegativeBinomial()
        cls.rvs = stats.nbinom.rvs
        cls.init()

    @pytest.mark.xfail(reason='Passing wrong number of args/kwargs to _parse_args_rvs', strict=True, raises=TypeError)
    def test_fitted(self):
        super().test_fitted()

    @pytest.mark.xfail(reason='Passing wrong number of args/kwargs to _parse_args_rvs', strict=True, raises=TypeError)
    def test_df(self):
        super().test_df()