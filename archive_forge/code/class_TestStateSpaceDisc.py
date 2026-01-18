import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
class TestStateSpaceDisc:

    def test_initialization(self):
        dt = 0.05
        StateSpace(1, 1, 1, 1, dt=dt)
        StateSpace([1], [2], [3], [4], dt=dt)
        StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]), np.array([[1, 0]]), np.array([[0]]), dt=dt)
        StateSpace(1, 1, 1, 1, dt=True)

    def test_conversion(self):
        s = StateSpace(1, 2, 3, 4, dt=0.05)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))
        assert_(StateSpace(s) is not s)
        assert_(s.to_ss() is not s)

    def test_properties(self):
        s = StateSpace(1, 1, 1, 1, dt=0.05)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])