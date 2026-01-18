import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
class TestSakurai:
    """
    Sakurai tests
    """

    def test_specific_shape(self):
        sak = Sakurai(6)
        assert_array_equal(sak.toarray(), sak(np.eye(6)))
        a = np.array([[5, -4, 1, 0, 0, 0], [-4, 6, -4, 1, 0, 0], [1, -4, 6, -4, 1, 0], [0, 1, -4, 6, -4, 1], [0, 0, 1, -4, 6, -4], [0, 0, 0, 1, -4, 5]])
        np.array_equal(a, sak.toarray())
        np.array_equal(sak.tosparse().toarray(), sak.toarray())
        ab = np.array([[1, 1, 1, 1, 1, 1], [-4, -4, -4, -4, -4, -4], [5, 6, 6, 6, 6, 5]])
        np.array_equal(ab, sak.tobanded())
        e = np.array([0.03922866, 0.56703972, 2.41789479, 5.97822974, 10.54287655, 14.45473055])
        np.array_equal(e, sak.eigenvalues())
        np.array_equal(e[:2], sak.eigenvalues(2))

    @pytest.mark.parametrize('dtype', ALLDTYPES)
    def test_linearoperator_shape_dtype(self, dtype):
        n = 7
        sak = Sakurai(n, dtype=dtype)
        assert sak.shape == (n, n)
        assert sak.dtype == dtype
        assert_array_equal(sak.toarray(), Sakurai(n).toarray().astype(dtype))
        assert_array_equal(sak.tosparse().toarray(), Sakurai(n).tosparse().toarray().astype(dtype))

    @pytest.mark.parametrize('dtype', ALLDTYPES)
    @pytest.mark.parametrize('argument_dtype', ALLDTYPES)
    def test_dot(self, dtype, argument_dtype):
        """ Test the dot-product for type preservation and consistency.
        """
        result_dtype = np.promote_types(argument_dtype, dtype)
        n = 5
        sak = Sakurai(n)
        x0 = np.arange(n)
        x1 = x0.reshape((-1, 1))
        x2 = np.arange(2 * n).reshape((n, 2))
        input_set = [x0, x1, x2]
        for x in input_set:
            y = sak.dot(x.astype(argument_dtype))
            assert x.shape == y.shape
            assert np.can_cast(y.dtype, result_dtype)
            if x.ndim == 2:
                ya = sak.toarray() @ x.astype(argument_dtype)
                np.array_equal(y, ya)
                assert np.can_cast(ya.dtype, result_dtype)
                ys = sak.tosparse() @ x.astype(argument_dtype)
                np.array_equal(y, ys)
                assert np.can_cast(ys.dtype, result_dtype)