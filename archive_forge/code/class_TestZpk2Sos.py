import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestZpk2Sos:

    @pytest.mark.parametrize('dt', 'fdgFDG')
    @pytest.mark.parametrize('pairing, analog', [('nearest', False), ('keep_odd', False), ('minimal', False), ('minimal', True)])
    def test_dtypes(self, dt, pairing, analog):
        z = np.array([-1, -1]).astype(dt)
        ct = dt.upper()
        p = np.array([0.57149 + 0.2936j, 0.57149 - 0.2936j]).astype(ct)
        k = np.array(1).astype(dt)
        sos = zpk2sos(z, p, k, pairing=pairing, analog=analog)
        sos2 = [[1, 2, 1, 1, -1.14298, 0.4128]]
        assert_array_almost_equal(sos, sos2, decimal=4)

    def test_basic(self):
        for pairing in ('nearest', 'keep_odd'):
            z = [-1, -1]
            p = [0.57149 + 0.2936j, 0.57149 - 0.2936j]
            k = 1
            sos = zpk2sos(z, p, k, pairing=pairing)
            sos2 = [[1, 2, 1, 1, -1.14298, 0.4128]]
            assert_array_almost_equal(sos, sos2, decimal=4)
            z = [1j, -1j]
            p = [0.9, -0.9, 0.7j, -0.7j]
            k = 1
            sos = zpk2sos(z, p, k, pairing=pairing)
            sos2 = [[1, 0, 1, 1, 0, +0.49], [1, 0, 0, 1, 0, -0.81]]
            assert_array_almost_equal(sos, sos2, decimal=4)
            z = []
            p = [0.8, -0.5 + 0.25j, -0.5 - 0.25j]
            k = 1.0
            sos = zpk2sos(z, p, k, pairing=pairing)
            sos2 = [[1.0, 0.0, 0.0, 1.0, 1.0, 0.3125], [1.0, 0.0, 0.0, 1.0, -0.8, 0.0]]
            assert_array_almost_equal(sos, sos2, decimal=4)
            z = [1.0, 1.0, 0.9j, -0.9j]
            p = [0.99 + 0.01j, 0.99 - 0.01j, 0.1 + 0.9j, 0.1 - 0.9j]
            k = 1
            sos = zpk2sos(z, p, k, pairing=pairing)
            sos2 = [[1, 0, 0.81, 1, -0.2, 0.82], [1, -2, 1, 1, -1.98, 0.9802]]
            assert_array_almost_equal(sos, sos2, decimal=4)
            z = [0.9 + 0.1j, 0.9 - 0.1j, -0.9]
            p = [0.75 + 0.25j, 0.75 - 0.25j, 0.9]
            k = 1
            sos = zpk2sos(z, p, k, pairing=pairing)
            if pairing == 'keep_odd':
                sos2 = [[1, -1.8, 0.82, 1, -1.5, 0.625], [1, 0.9, 0, 1, -0.9, 0]]
                assert_array_almost_equal(sos, sos2, decimal=4)
            else:
                sos2 = [[1, 0.9, 0, 1, -1.5, 0.625], [1, -1.8, 0.82, 1, -0.9, 0]]
                assert_array_almost_equal(sos, sos2, decimal=4)
            z = [-0.309 + 0.9511j, -0.309 - 0.9511j, 0.809 + 0.5878j, +0.809 - 0.5878j, -1.0 + 0j]
            p = [-0.3026 + 0.9312j, -0.3026 - 0.9312j, 0.7922 + 0.5755j, +0.7922 - 0.5755j, -0.9791 + 0j]
            k = 1
            sos = zpk2sos(z, p, k, pairing=pairing)
            sos2 = [[1, 1, 0, 1, +0.97915, 0], [1, 0.61803, 1, 1, +0.60515, 0.95873], [1, -1.61803, 1, 1, -1.5843, 0.95873]]
            assert_array_almost_equal(sos, sos2, decimal=4)
            z = [-1 - 1.4142j, -1 + 1.4142j, -0.625 - 1.0533j, -0.625 + 1.0533j]
            p = [-0.2 - 0.6782j, -0.2 + 0.6782j, -0.1 - 0.5385j, -0.1 + 0.5385j]
            k = 4
            sos = zpk2sos(z, p, k, pairing=pairing)
            sos2 = [[4, 8, 12, 1, 0.2, 0.3], [1, 1.25, 1.5, 1, 0.4, 0.5]]
            assert_allclose(sos, sos2, rtol=0.0001, atol=0.0001)
            z = []
            p = [0.2, -0.5 + 0.25j, -0.5 - 0.25j]
            k = 1.0
            sos = zpk2sos(z, p, k, pairing=pairing)
            sos2 = [[1.0, 0.0, 0.0, 1.0, -0.2, 0.0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.3125]]
            assert_array_almost_equal(sos, sos2, decimal=4)
            deg2rad = np.pi / 180.0
            k = 1.0
            thetas = [22.5, 45, 77.5]
            mags = [0.8, 0.6, 0.9]
            z = np.array([np.exp(theta * deg2rad * 1j) for theta in thetas])
            z = np.concatenate((z, np.conj(z)))
            p = np.array([mag * np.exp(theta * deg2rad * 1j) for theta, mag in zip(thetas, mags)])
            p = np.concatenate((p, np.conj(p)))
            sos = zpk2sos(z, p, k)
            sos2 = [[1, -1.41421, 1, 1, -0.84853, 0.36], [1, -1.84776, 1, 1, -1.47821, 0.64], [1, -0.43288, 1, 1, -0.38959, 0.81]]
            assert_array_almost_equal(sos, sos2, decimal=4)
            z = np.array([np.exp(theta * deg2rad * 1j) for theta in (85.0, 10.0)])
            z = np.concatenate((z, np.conj(z), [1, -1]))
            sos = zpk2sos(z, p, k)
            sos2 = [[1, 0, -1, 1, -0.84853, 0.36], [1, -1.96962, 1, 1, -1.47821, 0.64], [1, -0.17431, 1, 1, -0.38959, 0.81]]
            assert_array_almost_equal(sos, sos2, decimal=4)

    @pytest.mark.parametrize('pairing, sos', [('nearest', np.array([[1.0, 1.0, 0.5, 1.0, -0.75, 0.0], [1.0, 1.0, 0.0, 1.0, -1.6, 0.65]])), ('keep_odd', np.array([[1.0, 1.0, 0, 1.0, -0.75, 0.0], [1.0, 1.0, 0.5, 1.0, -1.6, 0.65]])), ('minimal', np.array([[0.0, 1.0, 1.0, 0.0, 1.0, -0.75], [1.0, 1.0, 0.5, 1.0, -1.6, 0.65]]))])
    def test_pairing(self, pairing, sos):
        z1 = np.array([-1, -0.5 - 0.5j, -0.5 + 0.5j])
        p1 = np.array([0.75, 0.8 + 0.1j, 0.8 - 0.1j])
        sos2 = zpk2sos(z1, p1, 1, pairing=pairing)
        assert_array_almost_equal(sos, sos2, decimal=4)

    @pytest.mark.parametrize('p, sos_dt', [([-1, 1, -0.1, 0.1], [[0.0, 0.0, 1.0, 1.0, 0.0, -0.01], [0.0, 0.0, 1.0, 1.0, 0.0, -1]]), ([-0.7071 + 0.7071j, -0.7071 - 0.7071j, -0.1j, 0.1j], [[0.0, 0.0, 1.0, 1.0, 0.0, 0.01], [0.0, 0.0, 1.0, 1.0, 1.4142, 1.0]])])
    def test_analog(self, p, sos_dt):
        sos2_dt = zpk2sos([], p, 1, pairing='minimal', analog=False)
        sos2_ct = zpk2sos([], p, 1, pairing='minimal', analog=True)
        assert_array_almost_equal(sos_dt, sos2_dt, decimal=4)
        assert_array_almost_equal(sos_dt[::-1], sos2_ct, decimal=4)

    def test_bad_args(self):
        with pytest.raises(ValueError, match='pairing must be one of'):
            zpk2sos([1], [2], 1, pairing='no_such_pairing')
        with pytest.raises(ValueError, match='.*pairing must be "minimal"'):
            zpk2sos([1], [2], 1, pairing='keep_odd', analog=True)
        with pytest.raises(ValueError, match='.*must have len\\(p\\)>=len\\(z\\)'):
            zpk2sos([1, 1], [2], 1, analog=True)
        with pytest.raises(ValueError, match='k must be real'):
            zpk2sos([1], [2], k=1j)