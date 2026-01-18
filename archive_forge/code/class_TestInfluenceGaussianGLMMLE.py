from statsmodels.compat.pandas import testing as pdt
import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import MLEInfluence
class TestInfluenceGaussianGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)), columns='y const var1 var2'.split(), index=labels)
        res = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)