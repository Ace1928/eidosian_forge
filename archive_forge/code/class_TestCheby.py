import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
class TestCheby:

    def test_chebyc(self):
        C0 = orth.chebyc(0)
        C1 = orth.chebyc(1)
        with np.errstate(all='ignore'):
            C2 = orth.chebyc(2)
            C3 = orth.chebyc(3)
            C4 = orth.chebyc(4)
            C5 = orth.chebyc(5)
        assert_array_almost_equal(C0.c, [2], 13)
        assert_array_almost_equal(C1.c, [1, 0], 13)
        assert_array_almost_equal(C2.c, [1, 0, -2], 13)
        assert_array_almost_equal(C3.c, [1, 0, -3, 0], 13)
        assert_array_almost_equal(C4.c, [1, 0, -4, 0, 2], 13)
        assert_array_almost_equal(C5.c, [1, 0, -5, 0, 5, 0], 13)

    def test_chebys(self):
        S0 = orth.chebys(0)
        S1 = orth.chebys(1)
        S2 = orth.chebys(2)
        S3 = orth.chebys(3)
        S4 = orth.chebys(4)
        S5 = orth.chebys(5)
        assert_array_almost_equal(S0.c, [1], 13)
        assert_array_almost_equal(S1.c, [1, 0], 13)
        assert_array_almost_equal(S2.c, [1, 0, -1], 13)
        assert_array_almost_equal(S3.c, [1, 0, -2, 0], 13)
        assert_array_almost_equal(S4.c, [1, 0, -3, 0, 1], 13)
        assert_array_almost_equal(S5.c, [1, 0, -4, 0, 3, 0], 13)

    def test_chebyt(self):
        T0 = orth.chebyt(0)
        T1 = orth.chebyt(1)
        T2 = orth.chebyt(2)
        T3 = orth.chebyt(3)
        T4 = orth.chebyt(4)
        T5 = orth.chebyt(5)
        assert_array_almost_equal(T0.c, [1], 13)
        assert_array_almost_equal(T1.c, [1, 0], 13)
        assert_array_almost_equal(T2.c, [2, 0, -1], 13)
        assert_array_almost_equal(T3.c, [4, 0, -3, 0], 13)
        assert_array_almost_equal(T4.c, [8, 0, -8, 0, 1], 13)
        assert_array_almost_equal(T5.c, [16, 0, -20, 0, 5, 0], 13)

    def test_chebyu(self):
        U0 = orth.chebyu(0)
        U1 = orth.chebyu(1)
        U2 = orth.chebyu(2)
        U3 = orth.chebyu(3)
        U4 = orth.chebyu(4)
        U5 = orth.chebyu(5)
        assert_array_almost_equal(U0.c, [1], 13)
        assert_array_almost_equal(U1.c, [2, 0], 13)
        assert_array_almost_equal(U2.c, [4, 0, -1], 13)
        assert_array_almost_equal(U3.c, [8, 0, -4, 0], 13)
        assert_array_almost_equal(U4.c, [16, 0, -12, 0, 1], 13)
        assert_array_almost_equal(U5.c, [32, 0, -32, 0, 6, 0], 13)