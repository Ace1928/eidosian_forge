import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
class TestDynamicGUFunc(TestCase):
    target = 'cpu'

    def test_dynamic_matmul(self):

        def check_matmul_gufunc(gufunc, A, B, C):
            Gold = np.matmul(A, B)
            gufunc(A, B, C)
            np.testing.assert_allclose(C, Gold, rtol=1e-05, atol=1e-08)
        gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)', target=self.target, is_dynamic=True)
        matrix_ct = 10
        Ai64 = np.arange(matrix_ct * 2 * 4, dtype=np.int64).reshape(matrix_ct, 2, 4)
        Bi64 = np.arange(matrix_ct * 4 * 5, dtype=np.int64).reshape(matrix_ct, 4, 5)
        Ci64 = np.arange(matrix_ct * 2 * 5, dtype=np.int64).reshape(matrix_ct, 2, 5)
        check_matmul_gufunc(gufunc, Ai64, Bi64, Ci64)
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
        C = np.arange(matrix_ct * 2 * 5, dtype=np.float32).reshape(matrix_ct, 2, 5)
        check_matmul_gufunc(gufunc, A, B, C)
        self.assertEqual(len(gufunc.types), 2)

    def test_dynamic_ufunc_like(self):

        def check_ufunc_output(gufunc, x):
            out = np.zeros(10, dtype=x.dtype)
            out_kw = np.zeros(10, dtype=x.dtype)
            gufunc(x, x, x, out)
            gufunc(x, x, x, out=out_kw)
            golden = x * x + x
            np.testing.assert_equal(out, golden)
            np.testing.assert_equal(out_kw, golden)
        gufunc = GUVectorize(axpy, '(), (), () -> ()', target=self.target, is_dynamic=True)
        x = np.arange(10, dtype=np.intp)
        check_ufunc_output(gufunc, x)

    def test_dynamic_scalar_output(self):
        """
        Note that scalar output is a 0-dimension array that acts as
        a pointer to the output location.
        """

        @guvectorize('(n)->()', target=self.target, nopython=True)
        def sum_row(inp, out):
            tmp = 0.0
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[()] = tmp
        self.assertTrue(sum_row.is_dynamic)
        inp = np.arange(30000, dtype=np.int32).reshape(10000, 3)
        out = np.zeros(10000, dtype=np.int32)
        sum_row(inp, out)
        for i in range(inp.shape[0]):
            self.assertEqual(out[i], inp[i].sum())
        msg = "Too few arguments for function 'sum_row'."
        with self.assertRaisesRegex(TypeError, msg):
            sum_row(inp)

    def test_axis(self):

        @guvectorize('(n)->(n)')
        def my_cumsum(x, res):
            acc = 0
            for i in range(x.shape[0]):
                acc += x[i]
                res[i] = acc
        x = np.ones((20, 30))
        expected = np.cumsum(x, axis=0)
        y = np.zeros_like(expected)
        my_cumsum(x, y, axis=0)
        np.testing.assert_equal(y, expected)
        out_kw = np.zeros_like(y)
        my_cumsum(x, out=out_kw, axis=0)
        np.testing.assert_equal(out_kw, expected)

    def test_gufunc_attributes(self):

        @guvectorize('(n)->(n)')
        def gufunc(x, res):
            acc = 0
            for i in range(x.shape[0]):
                acc += x[i]
                res[i] = acc
        attrs = ['signature', 'accumulate', 'at', 'outer', 'reduce', 'reduceat']
        for attr in attrs:
            contains = hasattr(gufunc, attr)
            self.assertTrue(contains, 'dynamic gufunc not exporting "%s"' % (attr,))
        a = np.array([1, 2, 3, 4])
        res = np.array([0, 0, 0, 0])
        gufunc(a, res)
        self.assertPreciseEqual(res, np.array([1, 3, 6, 10]))
        self.assertEqual(gufunc.signature, '(n)->(n)')
        with self.assertRaises(RuntimeError) as raises:
            gufunc.accumulate(a)
        self.assertEqual(str(raises.exception), 'Reduction not defined on ufunc with signature')
        with self.assertRaises(RuntimeError) as raises:
            gufunc.reduce(a)
        self.assertEqual(str(raises.exception), 'Reduction not defined on ufunc with signature')
        with self.assertRaises(RuntimeError) as raises:
            gufunc.reduceat(a, [0, 2])
        self.assertEqual(str(raises.exception), 'Reduction not defined on ufunc with signature')
        with self.assertRaises(TypeError) as raises:
            gufunc.outer(a, a)
        self.assertEqual(str(raises.exception), 'method outer is not allowed in ufunc with non-trivial signature')

    def test_gufunc_attributes2(self):

        @guvectorize('(),()->()')
        def add(x, y, res):
            res[0] = x + y
        self.assertIsNone(add.signature)
        a = np.array([1, 2, 3, 4])
        b = np.array([4, 3, 2, 1])
        res = np.array([0, 0, 0, 0])
        add(a, b, res)
        self.assertPreciseEqual(res, np.array([5, 5, 5, 5]))
        self.assertIsNone(add.signature)
        self.assertEqual(add.reduce(a), 10)
        self.assertPreciseEqual(add.accumulate(a), np.array([1, 3, 6, 10]))
        self.assertPreciseEqual(add.outer([0, 1], [1, 2]), np.array([[1, 2], [2, 3]]))
        self.assertPreciseEqual(add.reduceat(a, [0, 2]), np.array([3, 7]))
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2])
        add.at(x, [0, 1], y)
        self.assertPreciseEqual(x, np.array([2, 4, 3, 4]))