import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
class Test_RandomDisplacement:

    def setup_method(self):
        self.stepsize = 1.0
        self.displace = RandomDisplacement(stepsize=self.stepsize)
        self.N = 300000
        self.x0 = np.zeros([self.N])

    def test_random(self):
        x = self.displace(self.x0)
        v = (2.0 * self.stepsize) ** 2 / 12
        assert_almost_equal(np.mean(x), 0.0, 1)
        assert_almost_equal(np.var(x), v, 1)