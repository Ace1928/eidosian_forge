import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
class TestAdditiveSeasonal(AdditiveSeasonal):
    """
    Notes
    -----

    Standard errors are very good for the OPG and quite good for the complex
    step approximation cases.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(results_sarimax.wpi1_seasonal)

    def test_bse(self):
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-06)
        assert_allclose(self.result.bse[2:4], self.true['se_ma_opg'], atol=1e-05)

    def test_bse_approx(self):
        bse = self.result._cov_params_approx(approx_complex_step=True).diagonal() ** 0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=0.0001)
        assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=0.0001)

    def test_bse_oim(self):
        bse = self.result._cov_params_oim().diagonal() ** 0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=0.01)
        assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=0.1)