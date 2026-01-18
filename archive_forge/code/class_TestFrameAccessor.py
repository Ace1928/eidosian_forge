import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestFrameAccessor:

    def test_accessor_raises(self):
        df = pd.DataFrame({'A': [0, 1]})
        with pytest.raises(AttributeError, match='sparse'):
            df.sparse

    @pytest.mark.parametrize('format', ['csc', 'csr', 'coo'])
    @pytest.mark.parametrize('labels', [None, list(string.ascii_letters[:10])])
    @pytest.mark.parametrize('dtype', ['float64', 'int64'])
    def test_from_spmatrix(self, format, labels, dtype):
        sp_sparse = pytest.importorskip('scipy.sparse')
        sp_dtype = SparseDtype(dtype, np.array(0, dtype=dtype).item())
        mat = sp_sparse.eye(10, format=format, dtype=dtype)
        result = pd.DataFrame.sparse.from_spmatrix(mat, index=labels, columns=labels)
        expected = pd.DataFrame(np.eye(10, dtype=dtype), index=labels, columns=labels).astype(sp_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('format', ['csc', 'csr', 'coo'])
    def test_from_spmatrix_including_explicit_zero(self, format):
        sp_sparse = pytest.importorskip('scipy.sparse')
        mat = sp_sparse.random(10, 2, density=0.5, format=format)
        mat.data[0] = 0
        result = pd.DataFrame.sparse.from_spmatrix(mat)
        dtype = SparseDtype('float64', 0.0)
        expected = pd.DataFrame(mat.todense()).astype(dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('columns', [['a', 'b'], pd.MultiIndex.from_product([['A'], ['a', 'b']]), ['a', 'a']])
    def test_from_spmatrix_columns(self, columns):
        sp_sparse = pytest.importorskip('scipy.sparse')
        dtype = SparseDtype('float64', 0.0)
        mat = sp_sparse.random(10, 2, density=0.5)
        result = pd.DataFrame.sparse.from_spmatrix(mat, columns=columns)
        expected = pd.DataFrame(mat.toarray(), columns=columns).astype(dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('colnames', [('A', 'B'), (1, 2), (1, pd.NA), (0.1, 0.2), ('x', 'x'), (0, 0)])
    def test_to_coo(self, colnames):
        sp_sparse = pytest.importorskip('scipy.sparse')
        df = pd.DataFrame({colnames[0]: [0, 1, 0], colnames[1]: [1, 0, 0]}, dtype='Sparse[int64, 0]')
        result = df.sparse.to_coo()
        expected = sp_sparse.coo_matrix(np.asarray(df))
        assert (result != expected).nnz == 0

    @pytest.mark.parametrize('fill_value', [1, np.nan])
    def test_to_coo_nonzero_fill_val_raises(self, fill_value):
        pytest.importorskip('scipy')
        df = pd.DataFrame({'A': SparseArray([fill_value, fill_value, fill_value, 2], fill_value=fill_value), 'B': SparseArray([fill_value, 2, fill_value, fill_value], fill_value=fill_value)})
        with pytest.raises(ValueError, match='fill value must be 0'):
            df.sparse.to_coo()

    def test_to_coo_midx_categorical(self):
        sp_sparse = pytest.importorskip('scipy.sparse')
        midx = pd.MultiIndex.from_arrays([pd.CategoricalIndex(list('ab'), name='x'), pd.CategoricalIndex([0, 1], name='y')])
        ser = pd.Series(1, index=midx, dtype='Sparse[int]')
        result = ser.sparse.to_coo(row_levels=['x'], column_levels=['y'])[0]
        expected = sp_sparse.coo_matrix((np.array([1, 1]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2))
        assert (result != expected).nnz == 0

    def test_to_dense(self):
        df = pd.DataFrame({'A': SparseArray([1, 0], dtype=SparseDtype('int64', 0)), 'B': SparseArray([1, 0], dtype=SparseDtype('int64', 1)), 'C': SparseArray([1.0, 0.0], dtype=SparseDtype('float64', 0.0))}, index=['b', 'a'])
        result = df.sparse.to_dense()
        expected = pd.DataFrame({'A': [1, 0], 'B': [1, 0], 'C': [1.0, 0.0]}, index=['b', 'a'])
        tm.assert_frame_equal(result, expected)

    def test_density(self):
        df = pd.DataFrame({'A': SparseArray([1, 0, 2, 1], fill_value=0), 'B': SparseArray([0, 1, 1, 1], fill_value=0)})
        res = df.sparse.density
        expected = 0.75
        assert res == expected

    @pytest.mark.parametrize('dtype', ['int64', 'float64'])
    @pytest.mark.parametrize('dense_index', [True, False])
    def test_series_from_coo(self, dtype, dense_index):
        sp_sparse = pytest.importorskip('scipy.sparse')
        A = sp_sparse.eye(3, format='coo', dtype=dtype)
        result = pd.Series.sparse.from_coo(A, dense_index=dense_index)
        index = pd.MultiIndex.from_tuples([np.array([0, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32), np.array([2, 2], dtype=np.int32)])
        expected = pd.Series(SparseArray(np.array([1, 1, 1], dtype=dtype)), index=index)
        if dense_index:
            expected = expected.reindex(pd.MultiIndex.from_product(index.levels))
        tm.assert_series_equal(result, expected)

    def test_series_from_coo_incorrect_format_raises(self):
        sp_sparse = pytest.importorskip('scipy.sparse')
        m = sp_sparse.csr_matrix(np.array([[0, 1], [0, 0]]))
        with pytest.raises(TypeError, match='Expected coo_matrix. Got csr_matrix instead.'):
            pd.Series.sparse.from_coo(m)

    def test_with_column_named_sparse(self):
        df = pd.DataFrame({'sparse': pd.arrays.SparseArray([1, 2])})
        assert isinstance(df.sparse, pd.core.arrays.sparse.accessor.SparseFrameAccessor)