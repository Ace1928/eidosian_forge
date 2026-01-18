import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
class TestMikotaPair:
    """
    MikotaPair tests
    """
    tested_types = REAL_DTYPES + COMPLEX_DTYPES

    def test_specific_shape(self):
        n = 6
        mik = MikotaPair(n)
        mik_k = mik.k
        mik_m = mik.m
        assert_array_equal(mik_k.toarray(), mik_k(np.eye(n)))
        assert_array_equal(mik_m.toarray(), mik_m(np.eye(n)))
        k = np.array([[11, -5, 0, 0, 0, 0], [-5, 9, -4, 0, 0, 0], [0, -4, 7, -3, 0, 0], [0, 0, -3, 5, -2, 0], [0, 0, 0, -2, 3, -1], [0, 0, 0, 0, -1, 1]])
        np.array_equal(k, mik_k.toarray())
        np.array_equal(mik_k.tosparse().toarray(), k)
        kb = np.array([[0, -5, -4, -3, -2, -1], [11, 9, 7, 5, 3, 1]])
        np.array_equal(kb, mik_k.tobanded())
        minv = np.arange(1, n + 1)
        np.array_equal(np.diag(1.0 / minv), mik_m.toarray())
        np.array_equal(mik_m.tosparse().toarray(), mik_m.toarray())
        np.array_equal(1.0 / minv, mik_m.tobanded())
        e = np.array([1, 4, 9, 16, 25, 36])
        np.array_equal(e, mik.eigenvalues())
        np.array_equal(e[:2], mik.eigenvalues(2))

    @pytest.mark.parametrize('dtype', tested_types)
    def test_linearoperator_shape_dtype(self, dtype):
        n = 7
        mik = MikotaPair(n, dtype=dtype)
        mik_k = mik.k
        mik_m = mik.m
        assert mik_k.shape == (n, n)
        assert mik_k.dtype == dtype
        assert mik_m.shape == (n, n)
        assert mik_m.dtype == dtype
        mik_default_dtype = MikotaPair(n)
        mikd_k = mik_default_dtype.k
        mikd_m = mik_default_dtype.m
        assert mikd_k.shape == (n, n)
        assert mikd_k.dtype == np.float64
        assert mikd_m.shape == (n, n)
        assert mikd_m.dtype == np.float64
        assert_array_equal(mik_k.toarray(), mikd_k.toarray().astype(dtype))
        assert_array_equal(mik_k.tosparse().toarray(), mikd_k.tosparse().toarray().astype(dtype))

    @pytest.mark.parametrize('dtype', tested_types)
    @pytest.mark.parametrize('argument_dtype', ALLDTYPES)
    def test_dot(self, dtype, argument_dtype):
        """ Test the dot-product for type preservation and consistency.
        """
        result_dtype = np.promote_types(argument_dtype, dtype)
        n = 5
        mik = MikotaPair(n, dtype=dtype)
        mik_k = mik.k
        mik_m = mik.m
        x0 = np.arange(n)
        x1 = x0.reshape((-1, 1))
        x2 = np.arange(2 * n).reshape((n, 2))
        lo_set = [mik_k, mik_m]
        input_set = [x0, x1, x2]
        for lo in lo_set:
            for x in input_set:
                y = lo.dot(x.astype(argument_dtype))
                assert x.shape == y.shape
                assert np.can_cast(y.dtype, result_dtype)
                if x.ndim == 2:
                    ya = lo.toarray() @ x.astype(argument_dtype)
                    np.array_equal(y, ya)
                    assert np.can_cast(ya.dtype, result_dtype)
                    ys = lo.tosparse() @ x.astype(argument_dtype)
                    np.array_equal(y, ys)
                    assert np.can_cast(ys.dtype, result_dtype)