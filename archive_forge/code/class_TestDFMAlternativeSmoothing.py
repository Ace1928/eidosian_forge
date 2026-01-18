import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose
class TestDFMAlternativeSmoothing(TestDFM):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model._kalman_smoother.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model._kalman_smoother._smooth_method, SMOOTH_ALTERNATIVE)