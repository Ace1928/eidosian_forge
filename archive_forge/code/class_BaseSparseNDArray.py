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
class BaseSparseNDArray(NDArray):
    """The base class of an NDArray stored in a sparse storage format.

    See CSRNDArray and RowSparseNDArray for more details.
    """

    def __repr__(self):
        """Returns a string representation of the sparse array."""
        shape_info = 'x'.join(['%d' % x for x in self.shape])
        return '\n<%s %s @%s>' % (self.__class__.__name__, shape_info, self.context)

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return subtract(self, other)

    def __mul__(self, other):
        return multiply(self, other)

    def __div__(self, other):
        return divide(self, other)

    def __iadd__(self, other):
        raise NotImplementedError()

    def __isub__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        raise NotImplementedError()

    def __idiv__(self, other):
        raise NotImplementedError()

    def __itruediv__(self, other):
        raise NotImplementedError()

    def _sync_copyfrom(self, source_array):
        raise NotImplementedError()

    def _at(self, idx):
        raise NotSupportedForSparseNDArray(self._at, '[idx]', idx)

    def _slice(self, start, stop):
        raise NotSupportedForSparseNDArray(self._slice, None, start, stop)

    def reshape(self, *shape, **kwargs):
        raise NotSupportedForSparseNDArray(self.reshape, None, shape)

    @property
    def size(self):
        raise NotImplementedError()

    def _aux_type(self, i):
        """Data-type of the array's ith aux data.

        Returns
        -------
        numpy.dtype
            This BaseSparseNDArray's aux data type.
        """
        aux_type = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetAuxType(self.handle, i, ctypes.byref(aux_type)))
        return _DTYPE_MX_TO_NP[aux_type.value]

    @property
    def _num_aux(self):
        """The number of aux data used to help store the sparse ndarray.
        """
        return len(_STORAGE_AUX_TYPES[self.stype])

    @property
    def _aux_types(self):
        """The data types of the aux data for the BaseSparseNDArray.
        """
        aux_types = []
        num_aux = self._num_aux
        for i in range(num_aux):
            aux_types.append(self._aux_type(i))
        return aux_types

    def asnumpy(self):
        """Return a dense ``numpy.ndarray`` object with value copied from this array
        """
        return self.tostype('default').asnumpy()

    def astype(self, dtype, copy=True):
        """Return a copy of the array after casting to a specified type.

        Parameters
        ----------
        dtype : numpy.dtype or str
            The type of the returned array.
        copy : bool
            Default `True`. By default, astype always returns a newly
            allocated ndarray on the same context. If this is set to
            `False`, and the dtype requested is the same as the ndarray's
            dtype, the ndarray is returned instead of a copy.

        Examples
        --------
        >>> x = mx.nd.sparse.zeros('row_sparse', (2,3), dtype='float32')
        >>> y = x.astype('int32')
        >>> y.dtype
        <type 'numpy.int32'>
        """
        if not copy and np.dtype(dtype) == self.dtype:
            return self
        res = zeros(shape=self.shape, ctx=self.context, dtype=dtype, stype=self.stype)
        self.copyto(res)
        return res

    def copyto(self, other):
        """Copies the value of this array to another array.

        Parameters
        ----------
        other : NDArray or CSRNDArray or RowSparseNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray or RowSparseNDArray
            The copied array.
        """
        if isinstance(other, NDArray):
            if other.handle is self.handle:
                warnings.warn('You are attempting to copy an array to itself', RuntimeWarning)
                return False
            return _internal._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = _ndarray_cls(_new_alloc_handle(self.stype, self.shape, other, True, self.dtype, self._aux_types))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def check_format(self, full_check=True):
        """Check whether the NDArray format is valid.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, rigorous check, O(N) operations. Otherwise
            basic check, O(1) operations (default True).
        """
        check_call(_LIB.MXNDArraySyncCheckFormat(self.handle, ctypes.c_bool(full_check)))

    def _data(self):
        """A deep copy NDArray of the data array associated with the BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
        """
        self.wait_to_read()
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetDataNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl)

    def _aux_data(self, i):
        """ Get a deep copy NDArray of the i-th aux data array associated with the
        BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
        """
        self.wait_to_read()
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetAuxNDArray(self.handle, i, ctypes.byref(hdl)))
        return NDArray(hdl)