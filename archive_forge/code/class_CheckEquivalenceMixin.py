import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
class CheckEquivalenceMixin:
    tol = {'default': (0.0001, 1e-20)}

    @classmethod
    def get_sample(cls):
        np.random.seed(987456)
        nobs, k_vars = (200, 5)
        beta = 0.5 * np.array([0.1, 1, 1, 0, 0])
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        y = np.dot(x, beta) + 2 * np.random.randn(nobs)
        return (y, x)

    def test_attributes(self):
        attributes_fit = ['params', 'rsquared', 'df_resid', 'df_model', 'llf', 'aic', 'bic']
        attributes_inference = ['bse', 'tvalues', 'pvalues']
        import copy
        attributes = copy.copy(attributes_fit)
        if not getattr(self, 'skip_inference', False):
            attributes.extend(attributes_inference)
        for att in attributes:
            r1 = getattr(self.res1, att)
            r2 = getattr(self.res2, att)
            if not np.size(r1) == 1:
                r1 = r1[:len(r2)]
            rtol, atol = self.tol.get(att, self.tol['default'])
            message = 'attribute: ' + att
            assert_allclose(r1, r2, rtol=rtol, atol=atol, err_msg=message)
        assert_allclose(self.res1.fittedvalues, self.res1.fittedvalues, rtol=0.001, atol=0.0001)
        assert_allclose(self.res1.resid, self.res1.resid, rtol=0.001, atol=0.0001)