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
class Test64Bit:
    MAT_CLASSES = [bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dia_matrix]

    def _create_some_matrix(self, mat_cls, m, n):
        return mat_cls(np.random.rand(m, n))

    def _compare_index_dtype(self, m, dtype):
        dtype = np.dtype(dtype)
        if isinstance(m, (csc_matrix, csr_matrix, bsr_matrix)):
            return m.indices.dtype == dtype and m.indptr.dtype == dtype
        elif isinstance(m, coo_matrix):
            return m.row.dtype == dtype and m.col.dtype == dtype
        elif isinstance(m, dia_matrix):
            return m.offsets.dtype == dtype
        else:
            raise ValueError(f'matrix {m!r} has no integer indices')

    def test_decorator_maxval_limit(self):

        @with_64bit_maxval_limit(maxval_limit=10)
        def check(mat_cls):
            m = mat_cls(np.random.rand(10, 1))
            assert_(self._compare_index_dtype(m, np.int32))
            m = mat_cls(np.random.rand(11, 1))
            assert_(self._compare_index_dtype(m, np.int64))
        for mat_cls in self.MAT_CLASSES:
            check(mat_cls)

    def test_decorator_maxval_random(self):

        @with_64bit_maxval_limit(random=True)
        def check(mat_cls):
            seen_32 = False
            seen_64 = False
            for k in range(100):
                m = self._create_some_matrix(mat_cls, 9, 9)
                seen_32 = seen_32 or self._compare_index_dtype(m, np.int32)
                seen_64 = seen_64 or self._compare_index_dtype(m, np.int64)
                if seen_32 and seen_64:
                    break
            else:
                raise AssertionError('both 32 and 64 bit indices not seen')
        for mat_cls in self.MAT_CLASSES:
            check(mat_cls)

    def _check_resiliency(self, cls, method_name, **kw):

        @with_64bit_maxval_limit(**kw)
        def check(cls, method_name):
            instance = cls()
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
            try:
                getattr(instance, method_name)()
            finally:
                if hasattr(instance, 'teardown_method'):
                    instance.teardown_method()
        check(cls, method_name)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_limit_10(self, cls, method_name):
        self._check_resiliency(cls, method_name, maxval_limit=10)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_random(self, cls, method_name):
        self._check_resiliency(cls, method_name, random=True)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_all_32(self, cls, method_name):
        self._check_resiliency(cls, method_name, fixed_dtype=np.int32)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_resiliency_all_64(self, cls, method_name):
        self._check_resiliency(cls, method_name, fixed_dtype=np.int64)

    @pytest.mark.parametrize('cls,method_name', cases_64bit())
    def test_no_64(self, cls, method_name):
        self._check_resiliency(cls, method_name, assert_32bit=True)

    def test_downcast_intp(self):

        @with_64bit_maxval_limit(fixed_dtype=np.int64, downcast_maxval=1)
        def check_limited():
            a = csc_matrix([[1, 2], [3, 4], [5, 6]])
            assert_raises(AssertionError, a.getnnz, axis=1)
            assert_raises(AssertionError, a.sum, axis=0)
            a = csr_matrix([[1, 2, 3], [3, 4, 6]])
            assert_raises(AssertionError, a.getnnz, axis=0)
            a = coo_matrix([[1, 2, 3], [3, 4, 5]])
            assert_raises(AssertionError, a.getnnz, axis=0)

        @with_64bit_maxval_limit(fixed_dtype=np.int64)
        def check_unlimited():
            a = csc_matrix([[1, 2], [3, 4], [5, 6]])
            a.getnnz(axis=1)
            a.sum(axis=0)
            a = csr_matrix([[1, 2, 3], [3, 4, 6]])
            a.getnnz(axis=0)
            a = coo_matrix([[1, 2, 3], [3, 4, 5]])
            a.getnnz(axis=0)
        check_limited()
        check_unlimited()