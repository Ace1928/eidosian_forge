import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
class TestWhiten:

    @array_api_compatible
    def test_whiten(self, xp):
        desired = xp.asarray([[5.08738849, 2.97091878], [3.19909255, 0.6966058], [4.51041982, 0.02640918], [4.38567074, 0.95120889], [2.3219148, 1.63195503]])
        obs = xp.asarray([[0.9874451, 0.82766775], [0.62093317, 0.19406729], [0.87545741, 0.00735733], [0.85124403, 0.26499712], [0.4506759, 0.45464607]])
        xp_assert_close(whiten(obs), desired, rtol=1e-05)

    @array_api_compatible
    def test_whiten_zero_std(self, xp):
        desired = xp.asarray([[0.0, 1.0, 2.86666544], [0.0, 1.0, 1.32460034], [0.0, 1.0, 3.74382172]])
        obs = xp.asarray([[0.0, 1.0, 0.74109533], [0.0, 1.0, 0.34243798], [0.0, 1.0, 0.96785929]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            xp_assert_close(whiten(obs), desired, rtol=1e-05)
            assert_equal(len(w), 1)
            assert_(issubclass(w[-1].category, RuntimeWarning))

    @array_api_compatible
    def test_whiten_not_finite(self, xp):
        arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
        for tp in arrays:
            for bad_value in (xp.nan, xp.inf, -xp.inf):
                obs = tp([[0.9874451, bad_value], [0.62093317, 0.19406729], [0.87545741, 0.00735733], [0.85124403, 0.26499712], [0.4506759, 0.45464607]])
                assert_raises(ValueError, whiten, obs)