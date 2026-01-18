from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
class TestWLSExogWeights(CheckRegressionResults):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.ccard import load
        from .results.results_regression import CCardWLS
        dta = load()
        endog = np.asarray(dta.endog)
        exog = np.asarray(dta.exog)
        exog = add_constant(exog, prepend=False)
        nobs = 72.0
        weights = 1 / exog[:, 2]
        scaled_weights = weights * nobs / weights.sum()
        cls.res1 = WLS(endog, exog, weights=scaled_weights).fit()
        cls.res2 = CCardWLS()
        cls.res2.wresid = scaled_weights ** 0.5 * cls.res2.resid
        corr_ic = 2 * (cls.res1.llf - cls.res2.llf)
        cls.res2.aic -= corr_ic
        cls.res2.bic -= corr_ic
        cls.res2.llf += 0.5 * np.sum(np.log(cls.res1.model.weights))