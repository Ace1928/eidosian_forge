import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
class TestTheilLinRestrictionApprox(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        x2 = x[:, :2].copy()
        x2[:, 1] += x[:, 2]
        mod1 = TheilGLS(y, x[:, :3], r_matrix=[[0, 1, -1]])
        cls.res1 = mod1.fit(100)
        cls.res2 = OLS(y, x2).fit()
        import copy
        tol = copy.copy(cls.tol)
        tol2 = {'default': (0.15, 0), 'params': (0.05, 0), 'pvalues': (0.02, 0.001)}
        tol.update(tol2)
        cls.tol = tol