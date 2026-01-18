import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
class TestZerosPolesGain:

    def test_initialization(self):
        dt = 0.05
        ZerosPolesGain(1, 1, 1, dt=dt)
        ZerosPolesGain([1], [2], 1, dt=dt)
        ZerosPolesGain(np.array([1]), np.array([2]), 1, dt=dt)
        ZerosPolesGain(1, 1, 1, dt=True)

    def test_conversion(self):
        s = ZerosPolesGain(1, 2, 3, dt=0.05)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))
        assert_(ZerosPolesGain(s) is not s)
        assert_(s.to_zpk() is not s)