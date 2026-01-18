import os
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from statsmodels.tsa.statespace.sarimax import SARIMAX
class TestMultivariateMixedMissingGeneralObsCov(MultivariateMissingGeneralObsCov):
    """
    This class tests the univariate method when the observation covariance
    matrix is not diagonal and there are cases of both partially missing and
    fully missing data.

    Tests are against the conventional smoother.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class('mixed')

    def test_forecasts(self):
        assert_almost_equal(self.conventional_results.forecasts[0, :], self.univariate_results.forecasts[0, :], 8)

    def test_forecasts_error(self):
        assert_almost_equal(self.conventional_results.forecasts_error[0, :], self.univariate_results.forecasts_error[0, :], 8)