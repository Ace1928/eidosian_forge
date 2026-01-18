import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
class TestShChebyt:

    def test_sh_chebyt(self):
        psub = np.poly1d([2, -1])
        Ts0 = orth.sh_chebyt(0)
        Ts1 = orth.sh_chebyt(1)
        Ts2 = orth.sh_chebyt(2)
        Ts3 = orth.sh_chebyt(3)
        Ts4 = orth.sh_chebyt(4)
        Ts5 = orth.sh_chebyt(5)
        tse0 = orth.chebyt(0)(psub)
        tse1 = orth.chebyt(1)(psub)
        tse2 = orth.chebyt(2)(psub)
        tse3 = orth.chebyt(3)(psub)
        tse4 = orth.chebyt(4)(psub)
        tse5 = orth.chebyt(5)(psub)
        assert_array_almost_equal(Ts0.c, tse0.c, 13)
        assert_array_almost_equal(Ts1.c, tse1.c, 13)
        assert_array_almost_equal(Ts2.c, tse2.c, 13)
        assert_array_almost_equal(Ts3.c, tse3.c, 13)
        assert_array_almost_equal(Ts4.c, tse4.c, 12)
        assert_array_almost_equal(Ts5.c, tse5.c, 12)