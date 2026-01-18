from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
class TestTuckeyHSD2s(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        cls.endog = dta2['StressReduction'][3:29]
        cls.groups = dta2['Treatment'][3:29]
        cls.alpha = 0.01
        cls.setup_class_()
        tukeyhsd2s = np.array([1.8888888888888888, 0.888888888888889, -1, 0.2658549, -0.5908785, -2.587133, 3.511923, 2.368656, 0.5871331, 0.002837638, 0.150456, 0.1266072]).reshape(3, 4, order='F')
        cls.meandiff2 = tukeyhsd2s[:, 0]
        cls.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        cls.reject2 = pvals < 0.01