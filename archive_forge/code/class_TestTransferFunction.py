import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
class TestTransferFunction:

    def test_initialization(self):
        dt = 0.05
        TransferFunction(1, 1, dt=dt)
        TransferFunction([1], [2], dt=dt)
        TransferFunction(np.array([1]), np.array([2]), dt=dt)
        TransferFunction(1, 1, dt=True)

    def test_conversion(self):
        s = TransferFunction([1, 0], [1, -1], dt=0.05)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))
        assert_(TransferFunction(s) is not s)
        assert_(s.to_tf() is not s)

    def test_properties(self):
        s = TransferFunction([1, 0], [1, -1], dt=0.05)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])