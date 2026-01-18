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
class TestGLS_WLS_equivalence(TestOLS_GLS_WLS_equivalence):

    @classmethod
    def setup_class(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        y = data.endog
        x = data.exog
        n = y.shape[0]
        np.random.seed(5)
        w = np.random.uniform(0.5, 1, n)
        w_inv = 1.0 / w
        cls.results = []
        cls.results.append(WLS(y, x, w).fit())
        cls.results.append(WLS(y, x, 0.01 * w).fit())
        cls.results.append(GLS(y, x, 100 * w_inv).fit())
        cls.results.append(GLS(y, x, np.diag(0.1 * w_inv)).fit())