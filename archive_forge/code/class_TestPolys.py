import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
class TestPolys:
    """
    Check that the eval_* functions agree with the constructed polynomials

    """

    def check_poly(self, func, cls, param_ranges=[], x_range=[], nn=10, nparam=10, nx=10, rtol=1e-08):
        np.random.seed(1234)
        dataset = []
        for n in np.arange(nn):
            params = [a + (b - a) * np.random.rand(nparam) for a, b in param_ranges]
            params = np.asarray(params).T
            if not param_ranges:
                params = [0]
            for p in params:
                if param_ranges:
                    p = (n,) + tuple(p)
                else:
                    p = (n,)
                x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(nx)
                x[0] = x_range[0]
                x[1] = x_range[1]
                poly = np.poly1d(cls(*p).coef)
                z = np.c_[np.tile(p, (nx, 1)), x, poly(x)]
                dataset.append(z)
        dataset = np.concatenate(dataset, axis=0)

        def polyfunc(*p):
            p = (p[0].astype(np.dtype('long')),) + p[1:]
            return func(*p)
        with np.errstate(all='raise'):
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges) + 2)), -1, rtol=rtol)
            ds.check()

    def test_jacobi(self):
        self.check_poly(_ufuncs.eval_jacobi, orth.jacobi, param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1], rtol=1e-05)

    def test_sh_jacobi(self):
        self.check_poly(_ufuncs.eval_sh_jacobi, orth.sh_jacobi, param_ranges=[(1, 10), (0, 1)], x_range=[0, 1], rtol=1e-05)

    def test_gegenbauer(self):
        self.check_poly(_ufuncs.eval_gegenbauer, orth.gegenbauer, param_ranges=[(-0.499, 10)], x_range=[-1, 1], rtol=1e-07)

    def test_chebyt(self):
        self.check_poly(_ufuncs.eval_chebyt, orth.chebyt, param_ranges=[], x_range=[-1, 1])

    def test_chebyu(self):
        self.check_poly(_ufuncs.eval_chebyu, orth.chebyu, param_ranges=[], x_range=[-1, 1])

    def test_chebys(self):
        self.check_poly(_ufuncs.eval_chebys, orth.chebys, param_ranges=[], x_range=[-2, 2])

    def test_chebyc(self):
        self.check_poly(_ufuncs.eval_chebyc, orth.chebyc, param_ranges=[], x_range=[-2, 2])

    def test_sh_chebyt(self):
        with np.errstate(all='ignore'):
            self.check_poly(_ufuncs.eval_sh_chebyt, orth.sh_chebyt, param_ranges=[], x_range=[0, 1])

    def test_sh_chebyu(self):
        self.check_poly(_ufuncs.eval_sh_chebyu, orth.sh_chebyu, param_ranges=[], x_range=[0, 1])

    def test_legendre(self):
        self.check_poly(_ufuncs.eval_legendre, orth.legendre, param_ranges=[], x_range=[-1, 1])

    def test_sh_legendre(self):
        with np.errstate(all='ignore'):
            self.check_poly(_ufuncs.eval_sh_legendre, orth.sh_legendre, param_ranges=[], x_range=[0, 1])

    def test_genlaguerre(self):
        self.check_poly(_ufuncs.eval_genlaguerre, orth.genlaguerre, param_ranges=[(-0.99, 10)], x_range=[0, 100])

    def test_laguerre(self):
        self.check_poly(_ufuncs.eval_laguerre, orth.laguerre, param_ranges=[], x_range=[0, 100])

    def test_hermite(self):
        self.check_poly(_ufuncs.eval_hermite, orth.hermite, param_ranges=[], x_range=[-100, 100])

    def test_hermitenorm(self):
        self.check_poly(_ufuncs.eval_hermitenorm, orth.hermitenorm, param_ranges=[], x_range=[-100, 100])