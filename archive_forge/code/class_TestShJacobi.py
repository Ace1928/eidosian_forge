import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
class TestShJacobi:

    def test_sh_jacobi(self):

        def conv(n, p):
            return gamma(n + 1) * gamma(n + p) / gamma(2 * n + p)
        psub = np.poly1d([2, -1])
        q = 4 * np.random.random()
        p = q - 1 + 2 * np.random.random()
        G0 = orth.sh_jacobi(0, p, q)
        G1 = orth.sh_jacobi(1, p, q)
        G2 = orth.sh_jacobi(2, p, q)
        G3 = orth.sh_jacobi(3, p, q)
        G4 = orth.sh_jacobi(4, p, q)
        G5 = orth.sh_jacobi(5, p, q)
        ge0 = orth.jacobi(0, p - q, q - 1)(psub) * conv(0, p)
        ge1 = orth.jacobi(1, p - q, q - 1)(psub) * conv(1, p)
        ge2 = orth.jacobi(2, p - q, q - 1)(psub) * conv(2, p)
        ge3 = orth.jacobi(3, p - q, q - 1)(psub) * conv(3, p)
        ge4 = orth.jacobi(4, p - q, q - 1)(psub) * conv(4, p)
        ge5 = orth.jacobi(5, p - q, q - 1)(psub) * conv(5, p)
        assert_array_almost_equal(G0.c, ge0.c, 13)
        assert_array_almost_equal(G1.c, ge1.c, 13)
        assert_array_almost_equal(G2.c, ge2.c, 13)
        assert_array_almost_equal(G3.c, ge3.c, 13)
        assert_array_almost_equal(G4.c, ge4.c, 13)
        assert_array_almost_equal(G5.c, ge5.c, 13)