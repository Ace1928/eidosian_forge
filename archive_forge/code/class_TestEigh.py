import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
class TestEigh:

    def setup_class(self):
        np.random.seed(1234)

    def test_wrong_inputs(self):
        assert_raises(ValueError, eigh, np.ones([1, 2]))
        assert_raises(ValueError, eigh, np.ones([2, 2]), np.ones([2, 1]))
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([2, 2]))
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), type=4)
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_value=[1, 2], subset_by_index=[2, 4])
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
            assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_value=[1, 2], eigvals=[2, 4])
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[0, 4])
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
            assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), eigvals=[0, 4])
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[-2, 2])
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
            assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), eigvals=[-2, 2])
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[2, 0])
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
            assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[2, 0])
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_value=[2, 0])
        assert_raises(ValueError, eigh, np.ones([2, 2]), driver='wrong')
        assert_raises(ValueError, eigh, np.ones([3, 3]), None, driver='gvx')
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), driver='evr')
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), driver='gvd', subset_by_index=[1, 2])
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), driver='gvd', subset_by_index=[1, 2])

    def test_nonpositive_b(self):
        assert_raises(LinAlgError, eigh, np.ones([3, 3]), np.ones([3, 3]))

    def test_value_subsets(self):
        for ind, dt in enumerate(DTYPES):
            a = _random_hermitian_matrix(20, dtype=dt)
            w, v = eigh(a, subset_by_value=[-2, 2])
            assert_equal(v.shape[1], len(w))
            assert all((w > -2) & (w < 2))
            b = _random_hermitian_matrix(20, posdef=True, dtype=dt)
            w, v = eigh(a, b, subset_by_value=[-2, 2])
            assert_equal(v.shape[1], len(w))
            assert all((w > -2) & (w < 2))

    def test_eigh_integer(self):
        a = array([[1, 2], [2, 7]])
        b = array([[3, 1], [1, 5]])
        w, z = eigh(a)
        w, z = eigh(a, b)

    def test_eigh_of_sparse(self):
        import scipy.sparse
        a = scipy.sparse.identity(2).tocsc()
        b = np.atleast_2d(a)
        assert_raises(ValueError, eigh, a)
        assert_raises(ValueError, eigh, b)

    @pytest.mark.parametrize('dtype_', DTYPES)
    @pytest.mark.parametrize('driver', ('ev', 'evd', 'evr', 'evx'))
    def test_various_drivers_standard(self, driver, dtype_):
        a = _random_hermitian_matrix(n=20, dtype=dtype_)
        w, v = eigh(a, driver=driver)
        assert_allclose(a @ v - v * w, 0.0, atol=1000 * np.finfo(dtype_).eps, rtol=0.0)

    @pytest.mark.parametrize('type', (1, 2, 3))
    @pytest.mark.parametrize('driver', ('gv', 'gvd', 'gvx'))
    def test_various_drivers_generalized(self, driver, type):
        atol = np.spacing(5000.0)
        a = _random_hermitian_matrix(20)
        b = _random_hermitian_matrix(20, posdef=True)
        w, v = eigh(a=a, b=b, driver=driver, type=type)
        if type == 1:
            assert_allclose(a @ v - w * (b @ v), 0.0, atol=atol, rtol=0.0)
        elif type == 2:
            assert_allclose(a @ b @ v - v * w, 0.0, atol=atol, rtol=0.0)
        else:
            assert_allclose(b @ a @ v - v * w, 0.0, atol=atol, rtol=0.0)

    def test_eigvalsh_new_args(self):
        a = _random_hermitian_matrix(5)
        w = eigvalsh(a, subset_by_index=[1, 2])
        assert_equal(len(w), 2)
        w2 = eigvalsh(a, subset_by_index=[1, 2])
        assert_equal(len(w2), 2)
        assert_allclose(w, w2)
        b = np.diag([1, 1.2, 1.3, 1.5, 2])
        w3 = eigvalsh(b, subset_by_value=[1, 1.4])
        assert_equal(len(w3), 2)
        assert_allclose(w3, np.array([1.2, 1.3]))

    @pytest.mark.parametrize('method', [eigh, eigvalsh])
    def test_deprecation_warnings(self, method):
        with pytest.warns(DeprecationWarning, match="Keyword argument 'turbo'"):
            method(np.zeros((2, 2)), turbo=True)
        with pytest.warns(DeprecationWarning, match="Keyword argument 'eigvals'"):
            method(np.zeros((2, 2)), eigvals=[0, 1])
        with pytest.deprecated_call(match='use keyword arguments'):
            method(np.zeros((2, 2)), np.eye(2, 2), True)

    def test_deprecation_results(self):
        a = _random_hermitian_matrix(3)
        b = _random_hermitian_matrix(3, posdef=True)
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'turbo'")
            w_dep, v_dep = eigh(a, b, turbo=True)
        w, v = eigh(a, b, driver='gvd')
        assert_allclose(w_dep, w)
        assert_allclose(v_dep, v)
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'eigvals'")
            w_dep, v_dep = eigh(a, eigvals=[0, 1])
        w, v = eigh(a, subset_by_index=[0, 1])
        assert_allclose(w_dep, w)
        assert_allclose(v_dep, v)