import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
class TestTopLevel:

    def setup_method(self):
        self.data = sm.datasets.longley.load()
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.res = self.mod_fit.resid
        self.prbplt = ProbPlot(self.mod_fit.resid, dist=stats.t, distargs=(4,))
        self.other_array = np.random.normal(size=self.prbplt.data.shape)
        self.other_prbplot = ProbPlot(self.other_array)

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        qqplot(self.res, line='r')

    @pytest.mark.matplotlib
    def test_qqplot_pltkwargs(self, close_figures):
        qqplot(self.res, line='r', marker='d', markerfacecolor='cornflowerblue', markeredgecolor='white', alpha=0.5)

    @pytest.mark.matplotlib
    def test_qqplot_2samples_prob_plot_objects(self, close_figures):
        for line in ['r', 'q', '45', 's']:
            qqplot_2samples(self.prbplt, self.other_prbplot, line=line)

    @pytest.mark.matplotlib
    def test_qqplot_2samples_arrays(self, close_figures):
        for line in ['r', 'q', '45', 's']:
            qqplot_2samples(self.res, self.other_array, line=line)