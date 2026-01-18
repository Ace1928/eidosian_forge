from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
class _kde_subclass1(stats.gaussian_kde):

    def __init__(self, dataset):
        self.dataset = np.atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        self.covariance_factor = self.scotts_factor
        self._compute_covariance()