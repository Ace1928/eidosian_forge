import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
class TestCall:

    def test_call(self):
        poly = []
        for n in range(5):
            poly.extend([x.strip() for x in ('\n                orth.jacobi(%(n)d,0.3,0.9)\n                orth.sh_jacobi(%(n)d,0.3,0.9)\n                orth.genlaguerre(%(n)d,0.3)\n                orth.laguerre(%(n)d)\n                orth.hermite(%(n)d)\n                orth.hermitenorm(%(n)d)\n                orth.gegenbauer(%(n)d,0.3)\n                orth.chebyt(%(n)d)\n                orth.chebyu(%(n)d)\n                orth.chebyc(%(n)d)\n                orth.chebys(%(n)d)\n                orth.sh_chebyt(%(n)d)\n                orth.sh_chebyu(%(n)d)\n                orth.legendre(%(n)d)\n                orth.sh_legendre(%(n)d)\n                ' % dict(n=n)).split()])
        with np.errstate(all='ignore'):
            for pstr in poly:
                p = eval(pstr)
                assert_almost_equal(p(0.315), np.poly1d(p.coef)(0.315), err_msg=pstr)