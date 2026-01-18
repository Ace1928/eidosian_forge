import operator
import warnings
import numpy
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util
class csr_matrix(_compressed._compressed_sparse_matrix):
    """Compressed Sparse Row matrix.

    This can be instantiated in several ways.

    ``csr_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``csr_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocsr()``.
    ``csr_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.
    ``csr_matrix((data, (row, col)))``
        All ``data``, ``row`` and ``col`` are one-dimenaional
        :class:`cupy.ndarray`.
    ``csr_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
        :class:`scipy.sparse.csr_matrix`

    """
    format = 'csr'

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.csr_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=self._shape)

    def _convert_dense(self, x):
        m = dense2csr(x)
        return (m.data, m.indices, m.indptr)

    def _swap(self, x, y):
        return (x, y)

    def _add_sparse(self, other, alpha, beta):
        self.sum_duplicates()
        other = other.tocsr()
        other.sum_duplicates()
        if cusparse.check_availability('csrgeam2'):
            csrgeam = cusparse.csrgeam2
        elif cusparse.check_availability('csrgeam'):
            csrgeam = cusparse.csrgeam
        else:
            raise NotImplementedError
        return csrgeam(self, other, alpha, beta)

    def _comparison(self, other, op, op_name):
        if _util.isscalarlike(other):
            data = cupy.asarray(other, dtype=self.dtype).reshape(1)
            if numpy.isnan(data[0]):
                if op_name == '_ne_':
                    return csr_matrix(cupy.ones(self.shape, dtype=numpy.bool_))
                else:
                    return csr_matrix(self.shape, dtype=numpy.bool_)
            indices = cupy.zeros((1,), dtype=numpy.int32)
            indptr = cupy.arange(2, dtype=numpy.int32)
            other = csr_matrix((data, indices, indptr), shape=(1, 1))
            return binopt_csr(self, other, op_name)
        elif _util.isdense(other):
            return op(self.todense(), other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if op_name in ('_ne_', '_lt_', '_gt_'):
                return binopt_csr(self, other, op_name)
            warnings.warn('Comparing sparse matrices using ==, <=, and >= is inefficient, try using !=, <, or > instead.', SparseEfficiencyWarning)
            if op_name == '_eq_':
                opposite_op_name = '_ne_'
            elif op_name == '_le_':
                opposite_op_name = '_gt_'
            elif op_name == '_ge_':
                opposite_op_name = '_lt_'
            res = binopt_csr(self, other, opposite_op_name)
            out = cupy.logical_not(res.toarray())
            return csr_matrix(out)
        raise NotImplementedError

    def __eq__(self, other):
        return self._comparison(other, operator.eq, '_eq_')

    def __ne__(self, other):
        return self._comparison(other, operator.ne, '_ne_')

    def __lt__(self, other):
        return self._comparison(other, operator.lt, '_lt_')

    def __gt__(self, other):
        return self._comparison(other, operator.gt, '_gt_')

    def __le__(self, other):
        return self._comparison(other, operator.le, '_le_')

    def __ge__(self, other):
        return self._comparison(other, operator.ge, '_ge_')

    def __mul__(self, other):
        if cupy.isscalar(other):
            self.sum_duplicates()
            return self._with_data(self.data * other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('spgemm'):
                return cusparse.spgemm(self, other)
            elif cusparse.check_availability('csrgemm2'):
                return cusparse.csrgemm2(self, other)
            elif cusparse.check_availability('csrgemm'):
                return cusparse.csrgemm(self, other)
            else:
                raise NotImplementedError
        elif _csc.isspmatrix_csc(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm') and (not runtime.is_hip):
                return cusparse.csrgemm(self, other.T, transb=True)
            elif cusparse.check_availability('spgemm'):
                b = other.tocsr()
                b.sum_duplicates()
                return cusparse.spgemm(self, b)
            elif cusparse.check_availability('csrgemm2'):
                b = other.tocsr()
                b.sum_duplicates()
                return cusparse.csrgemm2(self, b)
            else:
                raise NotImplementedError
        elif _base.isspmatrix(other):
            return self * other.tocsr()
        elif _base.isdense(other):
            if other.ndim == 0:
                self.sum_duplicates()
                return self._with_data(self.data * other)
            elif other.ndim == 1:
                self.sum_duplicates()
                other = cupy.asfortranarray(other)
                is_cub_safe = self.indptr.data.mem.size > self.indptr.size * self.indptr.dtype.itemsize
                is_cub_safe &= cub._get_cuda_build_version() < 11000
                for accelerator in _accelerator.get_routine_accelerators():
                    if accelerator == _accelerator.ACCELERATOR_CUB and (not runtime.is_hip) and is_cub_safe and other.flags.c_contiguous:
                        return cub.device_csrmv(self.shape[0], self.shape[1], self.nnz, self.data, self.indptr, self.indices, other)
                if cusparse.check_availability('csrmvEx') and self.nnz > 0 and cusparse.csrmvExIsAligned(self, other):
                    csrmv = cusparse.csrmvEx
                elif cusparse.check_availability('csrmv'):
                    csrmv = cusparse.csrmv
                elif cusparse.check_availability('spmv'):
                    csrmv = cusparse.spmv
                else:
                    raise NotImplementedError
                return csrmv(self, other)
            elif other.ndim == 2:
                self.sum_duplicates()
                if cusparse.check_availability('csrmm2'):
                    csrmm = cusparse.csrmm2
                elif cusparse.check_availability('spmm'):
                    csrmm = cusparse.spmm
                else:
                    raise NotImplementedError
                return csrmm(self, cupy.asfortranarray(other))
            else:
                raise ValueError('could not interpret dimensions')
        else:
            return NotImplemented

    def __div__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        """Point-wise division by another matrix, vector or scalar"""
        if _util.isscalarlike(other):
            dtype = self.dtype
            if dtype == numpy.float32:
                dtype = numpy.float64
            dtype = cupy.result_type(dtype, other)
            d = cupy.reciprocal(other, dtype=dtype)
            return multiply_by_scalar(self, d)
        elif _util.isdense(other):
            other = cupy.atleast_2d(other)
            check_shape_for_pointwise_op(self.shape, other.shape)
            return self.todense() / other
        elif _base.isspmatrix(other):
            check_shape_for_pointwise_op(self.shape, other.shape, allow_broadcasting=False)
            dtype = numpy.promote_types(self.dtype, other.dtype)
            if dtype.char not in 'FD':
                dtype = numpy.promote_types(numpy.float64, dtype)
            self_dense = self.todense().astype(dtype, copy=False)
            return self_dense / other.todense()
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def diagonal(self, k=0):
        rows, cols = self.shape
        ylen = min(rows + min(k, 0), cols - max(k, 0))
        if ylen <= 0:
            return cupy.empty(0, dtype=self.dtype)
        self.sum_duplicates()
        y = cupy.empty(ylen, dtype=self.dtype)
        _cupy_csr_diagonal()(k, rows, cols, self.data, self.indptr, self.indices, y)
        return y

    def eliminate_zeros(self):
        """Removes zero entories in place."""
        compress = cusparse.csr2csr_compress(self, 0)
        self.data = compress.data
        self.indices = compress.indices
        self.indptr = compress.indptr

    def _maximum_minimum(self, other, cupy_op, op_name, dense_check):
        if _util.isscalarlike(other):
            other = cupy.asarray(other, dtype=self.dtype)
            if dense_check(other):
                dtype = self.dtype
                if dtype == numpy.float32:
                    dtype = numpy.float64
                elif dtype == numpy.complex64:
                    dtype = numpy.complex128
                dtype = cupy.result_type(dtype, other)
                other = other.astype(dtype, copy=False)
                new_array = cupy_op(self.todense(), other)
                return csr_matrix(new_array)
            else:
                self.sum_duplicates()
                new_data = cupy_op(self.data, other)
                return csr_matrix((new_data, self.indices, self.indptr), shape=self.shape, dtype=self.dtype)
        elif _util.isdense(other):
            self.sum_duplicates()
            other = cupy.atleast_2d(other)
            return cupy_op(self.todense(), other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return binopt_csr(self, other, op_name)
        raise NotImplementedError

    def maximum(self, other):
        return self._maximum_minimum(other, cupy.maximum, '_maximum_', lambda x: x > 0)

    def minimum(self, other):
        return self._maximum_minimum(other, cupy.minimum, '_minimum_', lambda x: x < 0)

    def multiply(self, other):
        """Point-wise multiplication by another matrix, vector or scalar"""
        if cupy.isscalar(other):
            return multiply_by_scalar(self, other)
        elif _util.isdense(other):
            self.sum_duplicates()
            other = cupy.atleast_2d(other)
            return multiply_by_dense(self, other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return multiply_by_csr(self, other)
        else:
            msg = 'expected scalar, dense matrix/vector or csr matrix'
            raise TypeError(msg)

    def setdiag(self, values, k=0):
        """Set diagonal or off-diagonal elements of the array."""
        rows, cols = self.shape
        row_st, col_st = (max(0, -k), max(0, k))
        x_len = min(rows - row_st, cols - col_st)
        if x_len <= 0:
            raise ValueError('k exceeds matrix dimensions')
        values = values.astype(self.dtype)
        if values.ndim == 0:
            x_data = cupy.full((x_len,), values, dtype=self.dtype)
        else:
            x_len = min(x_len, values.size)
            x_data = values[:x_len]
        x_indices = cupy.arange(col_st, col_st + x_len, dtype='i')
        x_indptr = cupy.zeros((rows + 1,), dtype='i')
        x_indptr[row_st:row_st + x_len + 1] = cupy.arange(x_len + 1, dtype='i')
        x_indptr[row_st + x_len + 1:] = x_len
        x_data -= self.diagonal(k=k)[:x_len]
        y = self + csr_matrix((x_data, x_indices, x_indptr), shape=self.shape)
        self.data = y.data
        self.indices = y.indices
        self.indptr = y.indptr

    def sort_indices(self):
        """Sorts the indices of this matrix *in place*.

        .. warning::
            Calling this function might synchronize the device.

        """
        if not self.has_sorted_indices:
            cusparse.csrsort(self)
            self.has_sorted_indices = True

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order ({'C', 'F', None}): Whether to store data in C (row-major)
                order or F (column-major) order. Default is C-order.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same matrix.

        .. seealso:: :meth:`scipy.sparse.csr_matrix.toarray`

        """
        order = 'C' if order is None else order.upper()
        if self.nnz == 0:
            return cupy.zeros(shape=self.shape, dtype=self.dtype, order=order)
        if self.dtype.char not in 'fdFD':
            return csr2dense(self, order)
        x = self.copy()
        x.has_canonical_format = False
        x.sum_duplicates()
        if cusparse.check_availability('sparseToDense') and (not runtime.is_hip or x.nnz > 0):
            y = cusparse.sparseToDense(x)
            if order == 'F':
                return y
            elif order == 'C':
                return cupy.ascontiguousarray(y)
            else:
                raise ValueError('order not understood')
        elif order == 'C':
            return cusparse.csc2dense(x.T).T
        elif order == 'F':
            return cusparse.csr2dense(x)
        else:
            raise ValueError('order not understood')

    def tobsr(self, blocksize=None, copy=False):
        raise NotImplementedError

    def tocoo(self, copy=False):
        """Converts the matrix to COOdinate format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupyx.scipy.sparse.coo_matrix: Converted matrix.

        """
        if copy:
            data = self.data.copy()
            indices = self.indices.copy()
        else:
            data = self.data
            indices = self.indices
        return cusparse.csr2coo(self, data, indices)

    def tocsc(self, copy=False):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in csr to csc conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        if cusparse.check_availability('csr2csc'):
            csr2csc = cusparse.csr2csc
        elif cusparse.check_availability('csr2cscEx2'):
            csr2csc = cusparse.csr2cscEx2
        else:
            raise NotImplementedError
        return csr2csc(self)

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, the method returns itself.
                Otherwise it makes a copy of the matrix.

        Returns:
            cupyx.scipy.sparse.csr_matrix: Converted matrix.

        """
        if copy:
            return self.copy()
        else:
            return self

    def _tocsx(self):
        """Inverts the format.
        """
        return self.tocsc()

    def todia(self, copy=False):
        raise NotImplementedError

    def todok(self, copy=False):
        raise NotImplementedError

    def tolil(self, copy=False):
        raise NotImplementedError

    def transpose(self, axes=None, copy=False):
        """Returns a transpose matrix.

        Args:
            axes: This option is not supported.
            copy (bool): If ``True``, a returned matrix shares no data.
                Otherwise, it shared data arrays as much as possible.

        Returns:
            cupyx.scipy.sparse.spmatrix: Transpose matrix.

        """
        if axes is not None:
            raise ValueError("Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
        shape = (self.shape[1], self.shape[0])
        trans = _csc.csc_matrix((self.data, self.indices, self.indptr), shape=shape, copy=copy)
        trans.has_canonical_format = self.has_canonical_format
        return trans

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).

        Args:
            i (integer): Row

        Returns:
            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single row
        """
        return self._major_slice(slice(i, i + 1), copy=True)

    def getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSR matrix (column vector).

        Args:
            i (integer): Column

        Returns:
            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single column
        """
        return self._minor_slice(slice(i, i + 1), copy=True)

    def _get_intXarray(self, row, col):
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_slice(col, copy=True)

    def _get_sliceXint(self, row, col):
        col = slice(col, col + 1)
        copy = row.step in (1, None)
        return self._major_slice(row)._minor_slice(col, copy=copy)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        col = slice(col, col + 1)
        return self._major_index_fancy(row)._minor_slice(col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            start, stop, step = col.indices(self.shape[1])
            cols = cupy.arange(start, stop, step, self.indices.dtype)
            return self._get_arrayXarray(row, cols)
        return self._major_index_fancy(row)._minor_slice(col)