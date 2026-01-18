import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
class TestRecurrence:
    """
    Check that the eval_* functions sig='ld->d' and 'dd->d' agree.

    """

    def check_poly(self, func, param_ranges=[], x_range=[], nn=10, nparam=10, nx=10, rtol=1e-08):
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
                kw = dict(sig=(len(p) + 1) * 'd' + '->d')
                z = np.c_[np.tile(p, (nx, 1)), x, func(*p + (x,), **kw)]
                dataset.append(z)
        dataset = np.concatenate(dataset, axis=0)

        def polyfunc(*p):
            p = (p[0].astype(int),) + p[1:]
            kw = dict(sig='l' + (len(p) - 1) * 'd' + '->d')
            return func(*p, **kw)
        with np.errstate(all='raise'):
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges) + 2)), -1, rtol=rtol)
            ds.check()

    def test_jacobi(self):
        self.check_poly(_ufuncs.eval_jacobi, param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1])

    def test_sh_jacobi(self):
        self.check_poly(_ufuncs.eval_sh_jacobi, param_ranges=[(1, 10), (0, 1)], x_range=[0, 1])

    def test_gegenbauer(self):
        self.check_poly(_ufuncs.eval_gegenbauer, param_ranges=[(-0.499, 10)], x_range=[-1, 1])

    def test_chebyt(self):
        self.check_poly(_ufuncs.eval_chebyt, param_ranges=[], x_range=[-1, 1])

    def test_chebyu(self):
        self.check_poly(_ufuncs.eval_chebyu, param_ranges=[], x_range=[-1, 1])

    def test_chebys(self):
        self.check_poly(_ufuncs.eval_chebys, param_ranges=[], x_range=[-2, 2])

    def test_chebyc(self):
        self.check_poly(_ufuncs.eval_chebyc, param_ranges=[], x_range=[-2, 2])

    def test_sh_chebyt(self):
        self.check_poly(_ufuncs.eval_sh_chebyt, param_ranges=[], x_range=[0, 1])

    def test_sh_chebyu(self):
        self.check_poly(_ufuncs.eval_sh_chebyu, param_ranges=[], x_range=[0, 1])

    def test_legendre(self):
        self.check_poly(_ufuncs.eval_legendre, param_ranges=[], x_range=[-1, 1])

    def test_sh_legendre(self):
        self.check_poly(_ufuncs.eval_sh_legendre, param_ranges=[], x_range=[0, 1])

    def test_genlaguerre(self):
        self.check_poly(_ufuncs.eval_genlaguerre, param_ranges=[(-0.99, 10)], x_range=[0, 100])

    def test_laguerre(self):
        self.check_poly(_ufuncs.eval_laguerre, param_ranges=[], x_range=[0, 100])

    def test_hermite(self):
        v = _ufuncs.eval_hermite(70, 1.0)
        a = -1.457076485701412e+60
        assert_allclose(v, a)