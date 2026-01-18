from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
class _kde_subclass4(stats.gaussian_kde):

    def covariance_factor(self):
        return 0.5 * self.silverman_factor()