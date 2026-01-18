import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
class TestExpmFrechet:

    def test_expm_frechet(self):
        M = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2], [0, 0, 5, 6]], dtype=float)
        A = np.array([[1, 2], [5, 6]], dtype=float)
        E = np.array([[3, 4], [7, 8]], dtype=float)
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm(M)[:2, 2:]
        for kwargs in ({}, {'method': 'SPS'}, {'method': 'blockEnlarge'}):
            observed_expm, observed_frechet = expm_frechet(A, E, **kwargs)
            assert_allclose(expected_expm, observed_expm)
            assert_allclose(expected_frechet, observed_frechet)

    def test_small_norm_expm_frechet(self):
        M_original = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2], [0, 0, 5, 6]], dtype=float)
        A_original = np.array([[1, 2], [5, 6]], dtype=float)
        E_original = np.array([[3, 4], [7, 8]], dtype=float)
        A_original_norm_1 = scipy.linalg.norm(A_original, 1)
        selected_m_list = [1, 3, 5, 7, 9, 11, 13, 15]
        m_neighbor_pairs = zip(selected_m_list[:-1], selected_m_list[1:])
        for ma, mb in m_neighbor_pairs:
            ell_a = scipy.linalg._expm_frechet.ell_table_61[ma]
            ell_b = scipy.linalg._expm_frechet.ell_table_61[mb]
            target_norm_1 = 0.5 * (ell_a + ell_b)
            scale = target_norm_1 / A_original_norm_1
            M = scale * M_original
            A = scale * A_original
            E = scale * E_original
            expected_expm = scipy.linalg.expm(A)
            expected_frechet = scipy.linalg.expm(M)[:2, 2:]
            observed_expm, observed_frechet = expm_frechet(A, E)
            assert_allclose(expected_expm, observed_expm)
            assert_allclose(expected_frechet, observed_frechet)

    def test_fuzz(self):
        rfuncs = (np.random.uniform, np.random.normal, np.random.standard_cauchy, np.random.exponential)
        ntests = 100
        for i in range(ntests):
            rfunc = random.choice(rfuncs)
            target_norm_1 = random.expovariate(1.0)
            n = random.randrange(2, 16)
            A_original = rfunc(size=(n, n))
            E_original = rfunc(size=(n, n))
            A_original_norm_1 = scipy.linalg.norm(A_original, 1)
            scale = target_norm_1 / A_original_norm_1
            A = scale * A_original
            E = scale * E_original
            M = np.vstack([np.hstack([A, E]), np.hstack([np.zeros_like(A), A])])
            expected_expm = scipy.linalg.expm(A)
            expected_frechet = scipy.linalg.expm(M)[:n, n:]
            observed_expm, observed_frechet = expm_frechet(A, E)
            assert_allclose(expected_expm, observed_expm, atol=5e-08)
            assert_allclose(expected_frechet, observed_frechet, atol=1e-07)

    def test_problematic_matrix(self):
        A = np.array([[1.50591997, 1.93537998], [0.41203263, 0.23443516]], dtype=float)
        E = np.array([[1.87864034, 2.07055038], [1.34102727, 0.67341123]], dtype=float)
        scipy.linalg.norm(A, 1)
        sps_expm, sps_frechet = expm_frechet(A, E, method='SPS')
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method='blockEnlarge')
        assert_allclose(sps_expm, blockEnlarge_expm)
        assert_allclose(sps_frechet, blockEnlarge_frechet)

    @pytest.mark.slow
    @pytest.mark.skip(reason='this test is deliberately slow')
    def test_medium_matrix(self):
        n = 1000
        A = np.random.exponential(size=(n, n))
        E = np.random.exponential(size=(n, n))
        sps_expm, sps_frechet = expm_frechet(A, E, method='SPS')
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method='blockEnlarge')
        assert_allclose(sps_expm, blockEnlarge_expm)
        assert_allclose(sps_frechet, blockEnlarge_frechet)