from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
class TestMixedLMSummaryRegularized(TestMixedLMSummary):

    @classmethod
    def setup_class(cls):
        pid = np.repeat([0, 1], 5)
        x0 = np.repeat([1], 10)
        x1 = [1, 5, 7, 3, 5, 1, 2, 6, 9, 8]
        x2 = [6, 2, 1, 0, 1, 4, 3, 8, 2, 1]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        df = pd.DataFrame({'y': y, 'pid': pid, 'x0': x0, 'x1': x1, 'x2': x2})
        endog = df['y'].values
        exog = df[['x0', 'x1', 'x2']].values
        groups = df['pid'].values
        cls.res = MixedLM(endog, exog, groups=groups).fit_regularized()