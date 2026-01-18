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
class TestInfluenceLogitCompare(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        mod = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']], family=families.Binomial())
        res = mod.fit(method='newton', tol=1e-10)
        from statsmodels.discrete.discrete_model import Logit
        mod2 = Logit(df['constrict'], df[['const', 'log_rate', 'log_volumne']])
        res2 = mod2.fit(method='newton', tol=1e-10)
        cls.infl1 = res.get_influence()
        cls.infl0 = res2.get_influence()