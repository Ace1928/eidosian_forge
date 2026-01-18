from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
class _kde_subclass2(stats.gaussian_kde):

    def __init__(self, dataset):
        self.covariance_factor = self.scotts_factor
        super().__init__(dataset)