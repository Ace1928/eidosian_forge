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
@pytest.mark.smoke
class CheckFEVD:
    fevd = None

    @pytest.mark.matplotlib
    def test_fevd_plot(self, close_figures):
        self.fevd.plot()

    def test_fevd_repr(self):
        self.fevd

    def test_fevd_summary(self):
        self.fevd.summary()

    @pytest.mark.xfail(reason='FEVD.cov() is not implemented', raises=NotImplementedError, strict=True)
    def test_fevd_cov(self):
        covs = self.fevd.cov()
        raise NotImplementedError