from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
class TestPlacePoles:

    def _check(self, A, B, P, **kwargs):
        """
        Perform the most common tests on the poles computed by place_poles
        and return the Bunch object for further specific tests
        """
        fsf = place_poles(A, B, P, **kwargs)
        expected, _ = np.linalg.eig(A - np.dot(B, fsf.gain_matrix))
        _assert_poles_close(expected, fsf.requested_poles)
        _assert_poles_close(expected, fsf.computed_poles)
        _assert_poles_close(P, fsf.requested_poles)
        return fsf

    def test_real(self):
        A = np.array([1.38, -0.2077, 6.715, -5.676, -0.5814, -4.29, 0, 0.675, 1.067, 4.273, -6.654, 5.893, 0.048, 4.273, 1.343, -2.104]).reshape(4, 4)
        B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0]).reshape(4, 2)
        P = np.array([-0.2, -0.5, -5.0566, -8.6659])
        self._check(A, B, P, method='KNV0')
        self._check(A, B, P, method='YT')
        with np.errstate(invalid='ignore'):
            self._check(A, B, (2, 2, 3, 3))

    def test_complex(self):
        A = np.array([[0, 7, 0, 0], [0, 0, 0, 7 / 3.0], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        P = np.array([-3, -1, -2 - 1j, -2 + 1j])
        with np.errstate(divide='ignore', invalid='ignore'):
            self._check(A, B, P)
        P = [0 - 1e-06j, 0 + 1e-06j, -10, 10]
        with np.errstate(divide='ignore', invalid='ignore'):
            self._check(A, B, P, maxiter=1000)
        A = np.array([-2148, -2902, -2267, -598, -1722, -1829, -165, -283, -2546, -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300, -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709, -291, -338, -153, -1804, -1106, -1168, -867, -2297]).reshape(6, 6)
        B = np.array([-108, -374, -524, -1285, -1232, -161, -1204, -672, -637, -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502, -952, -1374, -62, -964, -930, -939, -792, -756, -1437, -491, -1543, -686]).reshape(6, 5)
        P = [-25.0 - 29j, -25.0 + 29j, 31.0 - 42j, 31.0 + 42j, 33.0 - 41j, 33.0 + 41j]
        self._check(A, B, P)
        big_A = np.ones((11, 11)) - np.eye(11)
        big_B = np.ones((11, 10)) - np.diag([1] * 10, 1)[:, 1:]
        big_A[:6, :6] = A
        big_B[:6, :5] = B
        P = [-10, -20, -30, 40, 50, 60, 70, -20 - 5j, -20 + 5j, 5 + 3j, 5 - 3j]
        with np.errstate(divide='ignore', invalid='ignore'):
            self._check(big_A, big_B, P)
        P = [-10, -20, -30, -40, -50, -60, -70, -80, -90, -100]
        self._check(big_A[:-1, :-1], big_B[:-1, :-1], P)
        P = [-10 + 10j, -20 + 20j, -30 + 30j, -40 + 40j, -50 + 50j, -10 - 10j, -20 - 20j, -30 - 30j, -40 - 40j, -50 - 50j]
        self._check(big_A[:-1, :-1], big_B[:-1, :-1], P)
        A = np.array([0, 7, 0, 0, 0, 0, 0, 7 / 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 9]).reshape(5, 5)
        B = np.array([0, 0, 0, 0, 1, 0, 0, 1, 2, 3]).reshape(5, 2)
        P = np.array([-2, -3 + 1j, -3 - 1j, -1 + 1j, -1 - 1j])
        with np.errstate(divide='ignore', invalid='ignore'):
            place_poles(A, B, P)
        P = np.array([-2, -3, -4, -1 + 1j, -1 - 1j])
        with np.errstate(divide='ignore', invalid='ignore'):
            self._check(A, B, P)

    def test_tricky_B(self):
        A = np.array([1.38, -0.2077, 6.715, -5.676, -0.5814, -4.29, 0, 0.675, 1.067, 4.273, -6.654, 5.893, 0.048, 4.273, 1.343, -2.104]).reshape(4, 4)
        B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 4)
        P = np.array([-0.2, -0.5, -5.0566, -8.6659])
        fsf = self._check(A, B, P)
        assert_equal(fsf.rtol, np.nan)
        assert_equal(fsf.nb_iter, np.nan)
        P = np.array((-2 + 1j, -2 - 1j, -3, -2))
        fsf = self._check(A, B, P)
        assert_equal(fsf.rtol, np.nan)
        assert_equal(fsf.nb_iter, np.nan)
        B = B[:, 0].reshape(4, 1)
        P = np.array((-2 + 1j, -2 - 1j, -3, -2))
        fsf = self._check(A, B, P)
        assert_equal(fsf.rtol, 0)
        assert_equal(fsf.nb_iter, 0)

    def test_errors(self):
        A = np.array([0, 7, 0, 0, 0, 0, 0, 7 / 3.0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(4, 4)
        B = np.array([0, 0, 0, 0, 1, 0, 0, 1]).reshape(4, 2)
        assert_raises(ValueError, place_poles, A, B, (-2.1, -2.2, -2.3, -2.4), method='foo')
        assert_raises(ValueError, place_poles, A, B, np.array((-2.1, -2.2, -2.3, -2.4)).reshape(4, 1))
        assert_raises(ValueError, place_poles, A[:, :, np.newaxis], B, (-2.1, -2.2, -2.3, -2.4))
        assert_raises(ValueError, place_poles, A, B[:, :, np.newaxis], (-2.1, -2.2, -2.3, -2.4))
        assert_raises(ValueError, place_poles, A, B, (-2.1, -2.2, -2.3, -2.4, -3))
        assert_raises(ValueError, place_poles, A, B, (-2.1, -2.2, -2.3))
        assert_raises(ValueError, place_poles, A, B, (-2.1, -2.2, -2.3, -2.4), rtol=42)
        assert_raises(ValueError, place_poles, A, B, (-2.1, -2.2, -2.3, -2.4), maxiter=-42)
        assert_raises(ValueError, place_poles, A, B, (-2, -2, -2, -2))
        assert_raises(ValueError, place_poles, np.ones((4, 4)), np.ones((4, 2)), (1, 2, 3, 4))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fsf = place_poles(A, B, (-1, -2, -3, -4), rtol=1e-16, maxiter=42)
            assert_(len(w) == 1)
            assert_(issubclass(w[-1].category, UserWarning))
            assert_('Convergence was not reached after maxiter iterations' in str(w[-1].message))
            assert_equal(fsf.nb_iter, 42)
        assert_raises(ValueError, place_poles, A, B, (-2 + 1j, -2 - 1j, -2 + 3j, -2))
        assert_raises(ValueError, place_poles, A[:, :3], B, (-2, -3, -4, -5))
        assert_raises(ValueError, place_poles, A, B[:3, :], (-2, -3, -4, -5))
        assert_raises(ValueError, place_poles, A, B, (-2 + 1j, -2 - 1j, -2 + 3j, -2 - 3j), method='KNV0')