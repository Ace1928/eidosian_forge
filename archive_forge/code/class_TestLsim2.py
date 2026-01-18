from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
class TestLsim2(_TestLsimFuncs):
    digits_accuracy = 6

    def func(self, *args, **kwargs):
        with warns(DeprecationWarning, match='lsim2 is deprecated'):
            t, y, x = lsim2(*args, **kwargs)
        return (t, np.squeeze(y), np.squeeze(x))

    def test_integrator_nonequal_timestamp(self):
        t = np.array([0.0, 1.0, 1.0, 3.0])
        u = np.array([0.0, 0.0, 1.0, 1.0])
        system = ([1.0], [1.0, 0.0])
        tout, y, x = self.func(system, u, t, X0=[1.0])
        expected_x = np.maximum(1.0, tout)
        assert_almost_equal(x, expected_x)

    def test_integrator_nonequal_timestamp_kwarg(self):
        t = np.array([0.0, 1.0, 1.0, 1.1, 1.1, 2.0])
        u = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        system = ([1.0], [1.0, 0.0])
        tout, y, x = self.func(system, u, t, hmax=0.01)
        expected_x = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
        assert_almost_equal(x, expected_x)

    def test_default_arguments(self):
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y, x = self.func(system, X0=[1.0, 0.0])
        expected_x = (1.0 - tout) * np.exp(-tout)
        assert_almost_equal(x[:, 0], expected_x)