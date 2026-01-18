import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
class TestCOO(sparse_test_class(getset=False, slicing=False, slicing_assign=False, fancy_indexing=False, fancy_assign=False)):
    spcreator = coo_matrix
    math_dtypes = [np.int_, np.float64, np.complex128]

    def test_constructor1(self):
        row = array([2, 3, 1, 3, 0, 1, 3, 0, 2, 1, 2])
        col = array([0, 1, 0, 0, 1, 1, 2, 2, 2, 2, 1])
        data = array([6.0, 10.0, 3.0, 9.0, 1.0, 4.0, 11.0, 2.0, 8.0, 5.0, 7.0])
        coo = coo_matrix((data, (row, col)), (4, 3))
        assert_array_equal(arange(12).reshape(4, 3), coo.toarray())
        coo = coo_matrix(([2 ** 63 + 1, 1], ([0, 1], [0, 1])), dtype=np.uint64)
        dense = array([[2 ** 63 + 1, 0], [0, 1]], dtype=np.uint64)
        assert_array_equal(dense, coo.toarray())

    def test_constructor2(self):
        row = array([0, 1, 2, 2, 2, 2, 0, 0, 2, 2])
        col = array([0, 2, 0, 2, 1, 1, 1, 0, 0, 2])
        data = array([2, 9, -4, 5, 7, 0, -1, 2, 1, -5])
        coo = coo_matrix((data, (row, col)), (3, 3))
        mat = array([[4, -1, 0], [0, 0, 9], [-3, 7, 0]])
        assert_array_equal(mat, coo.toarray())

    def test_constructor3(self):
        coo = coo_matrix((4, 3))
        assert_array_equal(coo.shape, (4, 3))
        assert_array_equal(coo.row, [])
        assert_array_equal(coo.col, [])
        assert_array_equal(coo.data, [])
        assert_array_equal(coo.toarray(), zeros((4, 3)))

    def test_constructor4(self):
        mat = array([[0, 1, 0, 0], [7, 0, 3, 0], [0, 4, 0, 0]])
        coo = coo_matrix(mat)
        assert_array_equal(coo.toarray(), mat)
        mat = array([0, 1, 0, 0])
        coo = coo_matrix(mat)
        assert_array_equal(coo.toarray(), mat.reshape(1, -1))
        with pytest.raises(TypeError, match='object cannot be interpreted'):
            coo_matrix([0, 11, 22, 33], ([0, 1, 2, 3], [0, 0, 0, 0]))
        with pytest.raises(ValueError, match='inconsistent shapes'):
            coo_matrix([0, 11, 22, 33], shape=(4, 4))

    def test_constructor_data_ij_dtypeNone(self):
        data = [1]
        coo = coo_matrix((data, ([0], [0])), dtype=None)
        assert coo.dtype == np.array(data).dtype

    @pytest.mark.xfail(run=False, reason='COO does not have a __getitem__')
    def test_iterator(self):
        pass

    def test_todia_all_zeros(self):
        zeros = [[0, 0]]
        dia = coo_matrix(zeros).todia()
        assert_array_equal(dia.toarray(), zeros)

    def test_sum_duplicates(self):
        coo = coo_matrix((4, 3))
        coo.sum_duplicates()
        coo = coo_matrix(([1, 2], ([1, 0], [1, 0])))
        coo.sum_duplicates()
        assert_array_equal(coo.toarray(), [[2, 0], [0, 1]])
        coo = coo_matrix(([1, 2], ([1, 1], [1, 1])))
        coo.sum_duplicates()
        assert_array_equal(coo.toarray(), [[0, 0], [0, 3]])
        assert_array_equal(coo.row, [1])
        assert_array_equal(coo.col, [1])
        assert_array_equal(coo.data, [3])

    def test_todok_duplicates(self):
        coo = coo_matrix(([1, 1, 1, 1], ([0, 2, 2, 0], [0, 1, 1, 0])))
        dok = coo.todok()
        assert_array_equal(dok.toarray(), coo.toarray())

    def test_eliminate_zeros(self):
        data = array([1, 0, 0, 0, 2, 0, 3, 0])
        row = array([0, 0, 0, 1, 1, 1, 1, 1])
        col = array([1, 2, 3, 4, 5, 6, 7, 8])
        asp = coo_matrix((data, (row, col)), shape=(2, 10))
        bsp = asp.copy()
        asp.eliminate_zeros()
        assert_((asp.data != 0).all())
        assert_array_equal(asp.toarray(), bsp.toarray())

    def test_reshape_copy(self):
        arr = [[0, 10, 0, 0], [0, 0, 0, 0], [0, 20, 30, 40]]
        new_shape = (2, 6)
        x = coo_matrix(arr)
        y = x.reshape(new_shape)
        assert_(y.data is x.data)
        y = x.reshape(new_shape, copy=False)
        assert_(y.data is x.data)
        y = x.reshape(new_shape, copy=True)
        assert_(not np.may_share_memory(y.data, x.data))

    def test_large_dimensions_reshape(self):
        mat1 = coo_matrix(([1], ([3000000], [1000])), (3000001, 1001))
        mat2 = coo_matrix(([1], ([1000], [3000000])), (1001, 3000001))
        assert_((mat1.reshape((1001, 3000001), order='C') != mat2).nnz == 0)
        assert_((mat2.reshape((3000001, 1001), order='F') != mat1).nnz == 0)