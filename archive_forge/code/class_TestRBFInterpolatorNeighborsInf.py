import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
class TestRBFInterpolatorNeighborsInf(TestRBFInterpolatorNeighborsNone):

    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs, neighbors=np.inf)

    def test_equivalent_to_rbf_interpolator(self):
        seq = Halton(1, scramble=False, seed=np.random.RandomState())
        x = 3 * seq.random(50)
        xitp = 3 * seq.random(50)
        y = _1d_test_function(x)
        yitp1 = self.build(x, y)(xitp)
        yitp2 = RBFInterpolator(x, y)(xitp)
        assert_allclose(yitp1, yitp2, atol=1e-08)