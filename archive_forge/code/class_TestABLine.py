import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
class TestABLine:

    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        X = sm.add_constant(np.random.normal(0, 20, size=30))
        y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
        mod = sm.OLS(y, X).fit()
        cls.X = X
        cls.y = y
        cls.mod = mod

    @pytest.mark.matplotlib
    def test_abline_model(self, close_figures):
        fig = abline_plot(model_results=self.mod)
        ax = fig.axes[0]
        ax.scatter(self.X[:, 1], self.y)
        close_or_save(pdf, fig)

    @pytest.mark.matplotlib
    def test_abline_model_ax(self, close_figures):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:, 1], self.y)
        fig = abline_plot(model_results=self.mod, ax=ax)
        close_or_save(pdf, fig)

    @pytest.mark.matplotlib
    def test_abline_ab(self, close_figures):
        mod = self.mod
        intercept, slope = mod.params
        fig = abline_plot(intercept=intercept, slope=slope)
        close_or_save(pdf, fig)

    @pytest.mark.matplotlib
    def test_abline_ab_ax(self, close_figures):
        mod = self.mod
        intercept, slope = mod.params
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:, 1], self.y)
        fig = abline_plot(intercept=intercept, slope=slope, ax=ax)
        close_or_save(pdf, fig)

    @pytest.mark.matplotlib
    def test_abline_remove(self, close_figures):
        mod = self.mod
        intercept, slope = mod.params
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:, 1], self.y)
        abline_plot(intercept=intercept, slope=slope, ax=ax)
        abline_plot(intercept=intercept, slope=2 * slope, ax=ax)
        lines = ax.get_lines()
        lines.pop(0).remove()
        close_or_save(pdf, fig)