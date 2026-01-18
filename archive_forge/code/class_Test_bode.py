import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
class Test_bode:

    def test_manual(self):
        dt = 0.1
        system = TransferFunction(0.3, [1, -0.2], dt=dt)
        w = [0.1, 0.5, 1, np.pi]
        w2, mag, phase = dbode(system, w=w)
        expected_mag = [-8.5329, -8.8396, -9.6162, -12.0412]
        assert_almost_equal(mag, expected_mag, decimal=4)
        expected_phase = [-7.1575, -35.2814, -67.9809, -180.0]
        assert_almost_equal(phase, expected_phase, decimal=4)
        assert_equal(np.array(w) / dt, w2)

    def test_auto(self):
        system = TransferFunction(0.3, [1, -0.2], dt=0.1)
        w = np.array([0.1, 0.5, 1, np.pi])
        w2, mag, phase = dbode(system, w=w)
        jw = np.exp(w * 1j)
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_mag = 20.0 * np.log10(abs(y))
        assert_almost_equal(mag, expected_mag)
        expected_phase = np.rad2deg(np.angle(y))
        assert_almost_equal(phase, expected_phase)

    def test_range(self):
        dt = 0.1
        system = TransferFunction(0.3, [1, -0.2], dt=0.1)
        n = 10
        expected_w = np.linspace(0, np.pi, n, endpoint=False) / dt
        w, mag, phase = dbode(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_one(self):
        system = TransferFunction([1], [1, -1], dt=0.1)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message='divide by zero')
            sup.filter(RuntimeWarning, message='invalid value encountered')
            w, mag, phase = dbode(system, n=2)
        assert_equal(w[0], 0.0)

    def test_imaginary(self):
        system = TransferFunction([1], [1, 0, 100], dt=0.1)
        dbode(system, n=2)

    def test_error(self):
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dbode, system)