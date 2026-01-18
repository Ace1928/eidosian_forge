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
class TestTrivariateConventionalAlternate(TestTrivariateConventional):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, alternate_timing=True, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1