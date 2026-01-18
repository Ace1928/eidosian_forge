from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
class E1_Results:
    """
    Results from Lütkepohl (2005) using E2 dataset
    """

    def __init__(self):
        self.irf_stderr = np.array([[[0.125, 0.546, 0.664], [0.032, 0.139, 0.169], [0.026, 0.112, 0.136]], [[0.129, 0.547, 0.663], [0.032, 0.134, 0.163], [0.026, 0.108, 0.131]], [[0.084, 0.385, 0.479], [0.016, 0.079, 0.095], [0.016, 0.078, 0.103]]])
        self.cum_irf_stderr = np.array([[[0.125, 0.546, 0.664], [0.032, 0.139, 0.169], [0.026, 0.112, 0.136]], [[0.149, 0.631, 0.764], [0.044, 0.185, 0.224], [0.033, 0.14, 0.169]], [[0.099, 0.468, 0.555], [0.038, 0.17, 0.205], [0.033, 0.15, 0.185]]])
        self.lr_stderr = np.array([[0.134, 0.645, 0.808], [0.048, 0.23, 0.288], [0.043, 0.208, 0.26]])