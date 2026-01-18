import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
class TestPlottingPosition:

    def setup_method(self):
        self.N = 13
        self.data = np.arange(self.N)

    def do_test(self, alpha, beta):
        smpp = gofplots.plotting_pos(self.N, a=alpha, b=beta)
        sppp = stats.mstats.plotting_positions(self.data, alpha=alpha, beta=beta)
        nptest.assert_array_almost_equal(smpp, sppp, decimal=5)

    @pytest.mark.matplotlib
    def test_weibull(self, close_figures):
        self.do_test(0, 0)

    @pytest.mark.matplotlib
    def test_lininterp(self, close_figures):
        self.do_test(0, 1)

    @pytest.mark.matplotlib
    def test_piecewise(self, close_figures):
        self.do_test(0.5, 0.5)

    @pytest.mark.matplotlib
    def test_approx_med_unbiased(self, close_figures):
        self.do_test(1.0 / 3.0, 1.0 / 3.0)

    @pytest.mark.matplotlib
    def test_cunnane(self, close_figures):
        self.do_test(0.4, 0.4)