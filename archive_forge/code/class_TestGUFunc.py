import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
class TestGUFunc(TestCase):
    target = 'cpu'

    def check_matmul_gufunc(self, gufunc):
        matrix_ct = 1001
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
        C = gufunc(A, B)
        Gold = np.matmul(A, B)
        np.testing.assert_allclose(C, Gold, rtol=1e-05, atol=1e-08)

    def test_gufunc(self):
        gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', target=self.target)
        gufunc.add((float32[:, :], float32[:, :], float32[:, :]))
        gufunc = gufunc.build_ufunc()
        self.check_matmul_gufunc(gufunc)

    def test_guvectorize_decor(self):
        gufunc = guvectorize([void(float32[:, :], float32[:, :], float32[:, :])], '(m,n),(n,p)->(m,p)', target=self.target)(matmulcore)
        self.check_matmul_gufunc(gufunc)

    def test_ufunc_like(self):
        gufunc = GUVectorize(axpy, '(), (), () -> ()', target=self.target)
        gufunc.add('(intp, intp, intp, intp[:])')
        gufunc = gufunc.build_ufunc()
        x = np.arange(10, dtype=np.intp)
        out = gufunc(x, x, x)
        np.testing.assert_equal(out, x * x + x)

    def test_axis(self):

        @guvectorize(['f8[:],f8[:]'], '(n)->(n)')
        def my_cumsum(x, res):
            acc = 0
            for i in range(x.shape[0]):
                acc += x[i]
                res[i] = acc
        x = np.ones((20, 30))
        y = my_cumsum(x, axis=0)
        expected = np.cumsum(x, axis=0)
        np.testing.assert_equal(y, expected)
        out_kw = np.zeros_like(y)
        my_cumsum(x, out=out_kw, axis=0)
        np.testing.assert_equal(out_kw, expected)

    def test_docstring(self):

        @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
        def gufunc(x, y, res):
            """docstring for gufunc"""
            for i in range(x.shape[0]):
                res[i] = x[i] + y
        self.assertEqual('numba.tests.npyufunc.test_gufunc', gufunc.__module__)
        self.assertEqual('gufunc', gufunc.__name__)
        self.assertEqual('TestGUFunc.test_docstring.<locals>.gufunc', gufunc.__qualname__)
        self.assertEqual('docstring for gufunc', gufunc.__doc__)