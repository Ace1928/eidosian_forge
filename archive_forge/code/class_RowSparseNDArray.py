import ctypes
import warnings
import operator
from array import array as native_array
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import NDArrayHandle, check_call
from ..context import Context, current_context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _int64_enabled
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper
class RowSparseNDArray(BaseSparseNDArray):
    """A sparse representation of a set of NDArray row slices at given indices.

    A RowSparseNDArray represents a multidimensional NDArray using two separate arrays: `data` and
    `indices`. The number of dimensions has to be at least 2.

    - data: an NDArray of any dtype with shape [D0, D1, ..., Dn].
    - indices: a 1-D int64 NDArray with shape [D0] with values sorted in ascending order.

    The `indices` stores the indices of the row slices with non-zeros,
    while the values are stored in `data`. The corresponding NDArray ``dense``
    represented by RowSparseNDArray ``rsp`` has

    ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

        >>> dense.asnumpy()
        array([[ 1.,  2., 3.],
               [ 0.,  0., 0.],
               [ 4.,  0., 5.],
               [ 0.,  0., 0.],
               [ 0.,  0., 0.]], dtype=float32)
        >>> rsp = dense.tostype('row_sparse')
        >>> rsp.indices.asnumpy()
        array([0, 2], dtype=int64)
        >>> rsp.data.asnumpy()
        array([[ 1.,  2., 3.],
               [ 4.,  0., 5.]], dtype=float32)

    A RowSparseNDArray is typically used to represent non-zero row slices of a large NDArray
    of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

    RowSparseNDArray is used principally in the definition of gradients for operations
    that have sparse gradients (e.g. sparse dot and sparse embedding).

    See Also
    --------
    row_sparse_array: Several ways to construct a RowSparseNDArray
    """

    def __reduce__(self):
        return (RowSparseNDArray, (None,), super(RowSparseNDArray, self).__getstate__())

    def __iadd__(self, other):
        (self + other).copyto(self)
        return self

    def __isub__(self, other):
        (self - other).copyto(self)
        return self

    def __imul__(self, other):
        (self * other).copyto(self)
        return self

    def __idiv__(self, other):
        (self / other).copyto(self)
        return self

    def __itruediv__(self, other):
        (self / other).copyto(self)
        return self

    def __getitem__(self, key):
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of this array.

        Parameters
        ----------
        key : mxnet.ndarray.NDArray.slice
            Indexing key.

        Examples
        --------
        >>> x = mx.nd.sparse.zeros('row_sparse', (2, 3))
        >>> x[:].asnumpy()
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
        """
        if isinstance(key, int):
            raise Exception('__getitem__ with int key is not implemented for RowSparseNDArray yet')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise Exception('RowSparseNDArray only supports [:] for __getitem__')
            return self
        if isinstance(key, tuple):
            raise ValueError('Multi-dimension indexing is not supported')
        raise ValueError('Undefined behaviour for {}'.format(key))

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value. Only slice key [:] is supported.

        Parameters
        ----------
        key : mxnet.ndarray.NDArray.slice
            The indexing key.
        value : NDArray or numpy.ndarray
            The value to set.

        Examples
        --------
        >>> src = mx.nd.row_sparse([[1, 0, 2], [4, 5, 6]], [0, 2], (3,3))
        >>> src.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  0.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> # assign RowSparseNDArray with same storage type
        >>> x = mx.nd.sparse.zeros('row_sparse', (3,3))
        >>> x[:] = src
        >>> x.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  0.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> # assign NDArray to RowSparseNDArray
        >>> x[:] = mx.nd.ones((3,3))
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly RowSparseNDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slice for RowSparseNDArray is not implmented yet.')
            if isinstance(value, NDArray):
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                _internal._set_value(float(value), out=self)
            elif isinstance(value, (np.ndarray, np.generic)):
                warnings.warn('Assigning non-NDArray object to RowSparseNDArray is not efficient', RuntimeWarning)
                tmp = _array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert isinstance(key, (int, tuple))
            raise TypeError('RowSparseNDArray only supports [:] for assignment')

    @property
    def indices(self):
        """A deep copy NDArray of the indices array of the RowSparseNDArray.
        This generates a deep copy of the row indices of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This RowSparseNDArray's indices array.
        """
        return self._aux_data(0)

    @property
    def data(self):
        """A deep copy NDArray of the data array of the RowSparseNDArray.
        This generates a deep copy of the `data` of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This RowSparseNDArray's data array.
        """
        return self._data()

    @indices.setter
    def indices(self, indices):
        raise NotImplementedError()

    @data.setter
    def data(self, data):
        raise NotImplementedError()

    def tostype(self, stype):
        """Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or RowSparseNDArray
            A copy of the array with the chosen storage stype
        """
        if stype == 'csr':
            raise ValueError('cast_storage from row_sparse to csr is not supported')
        return op.cast_storage(self, stype=stype)

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` or ``RowSparseNDArray`` object, then ``other.shape``
        and ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``RowSparseNDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or RowSparseNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or RowSparseNDArray
            The copied array. If ``other`` is an ``NDArray`` or ``RowSparseNDArray``, then the
            return value and ``other`` will point to the same ``NDArray`` or ``RowSparseNDArray``.
        """
        if isinstance(other, Context):
            return super(RowSparseNDArray, self).copyto(other)
        elif isinstance(other, NDArray):
            stype = other.stype
            if stype in ('default', 'row_sparse'):
                return super(RowSparseNDArray, self).copyto(other)
            else:
                raise TypeError('copyto does not support destination NDArray stype ' + str(stype))
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def retain(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`retain`.

        The arguments are the same as for :py:func:`retain`, with
        this array as data.
        """
        if not gs_retain:
            raise ImportError('gen_sparse could not be imported')
        return gs_retain(*args, **kwargs)