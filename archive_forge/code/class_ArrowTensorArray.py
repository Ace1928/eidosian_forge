import itertools
import json
import sys
from typing import Iterable, Optional, Tuple, List, Sequence, Union
from pkg_resources._vendor.packaging.version import parse as parse_version
import numpy as np
import pyarrow as pa
from ray.air.util.tensor_extensions.utils import (
from ray._private.utils import _get_pyarrow_version
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
class ArrowTensorArray(_ArrowTensorScalarIndexingMixin, pa.ExtensionArray):
    """
    An array of fixed-shape, homogeneous-typed tensors.

    This is the Arrow side of TensorArray.

    See Arrow docs for customizing extension arrays:
    https://arrow.apache.org/docs/python/extending_types.html#custom-extension-array-class
    """
    OFFSET_DTYPE = np.int32

    @classmethod
    def from_numpy(cls, arr: Union[np.ndarray, Iterable[np.ndarray]]) -> Union['ArrowTensorArray', 'ArrowVariableShapedTensorArray']:
        """
        Convert an ndarray or an iterable of ndarrays to an array of homogeneous-typed
        tensors. If given fixed-shape tensor elements, this will return an
        ``ArrowTensorArray``; if given variable-shape tensor elements, this will return
        an ``ArrowVariableShapedTensorArray``.

        Args:
            arr: An ndarray or an iterable of ndarrays.

        Returns:
            - If fixed-shape tensor elements, an ``ArrowTensorArray`` containing
              ``len(arr)`` tensors of fixed shape.
            - If variable-shaped tensor elements, an ``ArrowVariableShapedTensorArray``
              containing ``len(arr)`` tensors of variable shape.
            - If scalar elements, a ``pyarrow.Array``.
        """
        if isinstance(arr, (list, tuple)) and arr and isinstance(arr[0], np.ndarray):
            try:
                arr = np.stack(arr, axis=0)
            except ValueError:
                arr = np.array(arr, dtype=object)
        if isinstance(arr, np.ndarray):
            if len(arr) > 0 and np.isscalar(arr[0]):
                return pa.array(arr)
            if _is_ndarray_variable_shaped_tensor(arr):
                return ArrowVariableShapedTensorArray.from_numpy(arr)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            pa_dtype = pa.from_numpy_dtype(arr.dtype)
            if pa.types.is_string(pa_dtype):
                if arr.dtype.byteorder == '>' or (arr.dtype.byteorder == '=' and sys.byteorder == 'big'):
                    raise ValueError(f'Only little-endian string tensors are supported, but got: {arr.dtype}')
                pa_dtype = pa.binary(arr.dtype.itemsize)
            outer_len = arr.shape[0]
            element_shape = arr.shape[1:]
            total_num_items = arr.size
            num_items_per_element = np.prod(element_shape) if element_shape else 1
            if pa.types.is_boolean(pa_dtype):
                arr = np.packbits(arr, bitorder='little')
            data_buffer = pa.py_buffer(arr)
            data_array = pa.Array.from_buffers(pa_dtype, total_num_items, [None, data_buffer])
            offset_buffer = pa.py_buffer(cls.OFFSET_DTYPE([i * num_items_per_element for i in range(outer_len + 1)]))
            storage = pa.Array.from_buffers(pa.list_(pa_dtype), outer_len, [None, offset_buffer], children=[data_array])
            type_ = ArrowTensorType(element_shape, pa_dtype)
            return pa.ExtensionArray.from_storage(type_, storage)
        elif isinstance(arr, Iterable):
            return cls.from_numpy(list(arr))
        else:
            raise ValueError('Must give ndarray or iterable of ndarrays.')

    def _to_numpy(self, index: Optional[int]=None, zero_copy_only: bool=False):
        """
        Helper for getting either an element of the array of tensors as an
        ndarray, or the entire array of tensors as a single ndarray.

        Args:
            index: The index of the tensor element that we wish to return as
                an ndarray. If not given, the entire array of tensors is
                returned as an ndarray.
            zero_copy_only: If True, an exception will be raised if the
                conversion to a NumPy array would require copying the
                underlying data (e.g. in presence of nulls, or for
                non-primitive types). This argument is currently ignored, so
                zero-copy isn't enforced even if this argument is true.

        Returns:
            The corresponding tensor element as an ndarray if an index was
            given, or the entire array of tensors as an ndarray otherwise.
        """
        buffers = self.buffers()
        data_buffer = buffers[3]
        storage_list_type = self.storage.type
        value_type = storage_list_type.value_type
        ext_dtype = value_type.to_pandas_dtype()
        shape = self.type.shape
        if pa.types.is_boolean(value_type):
            buffer_item_width = value_type.bit_width
        else:
            buffer_item_width = value_type.bit_width // 8
        num_items_per_element = np.prod(shape) if shape else 1
        buffer_offset = self.offset * num_items_per_element
        offset = buffer_item_width * buffer_offset
        if index is not None:
            offset_buffer = buffers[1]
            offset_array = np.ndarray((len(self),), buffer=offset_buffer, dtype=self.OFFSET_DTYPE)
            index_offset = offset_array[index]
            offset += buffer_item_width * index_offset
        else:
            shape = (len(self),) + shape
        if pa.types.is_boolean(value_type):
            byte_bucket_offset = offset // 8
            bool_offset = offset % 8
            num_boolean_byte_buckets = 1 + (bool_offset + np.prod(shape) - 1) // 8
            arr = np.ndarray((num_boolean_byte_buckets,), dtype=np.uint8, buffer=data_buffer, offset=byte_bucket_offset)
            arr = np.unpackbits(arr, bitorder='little')
            return np.ndarray(shape, dtype=np.bool_, buffer=arr, offset=bool_offset)
        if pa.types.is_fixed_size_binary(value_type):
            ext_dtype = np.dtype(f'<U{value_type.byte_width // NUM_BYTES_PER_UNICODE_CHAR}')
        return np.ndarray(shape, dtype=ext_dtype, buffer=data_buffer, offset=offset)

    def to_numpy(self, zero_copy_only: bool=True):
        """
        Convert the entire array of tensors into a single ndarray.

        Args:
            zero_copy_only: If True, an exception will be raised if the
                conversion to a NumPy array would require copying the
                underlying data (e.g. in presence of nulls, or for
                non-primitive types). This argument is currently ignored, so
                zero-copy isn't enforced even if this argument is true.

        Returns:
            A single ndarray representing the entire array of tensors.
        """
        return self._to_numpy(zero_copy_only=zero_copy_only)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Union['ArrowTensorArray', 'ArrowVariableShapedTensorArray']]) -> Union['ArrowTensorArray', 'ArrowVariableShapedTensorArray']:
        """
        Concatenate multiple tensor arrays.

        If one or more of the tensor arrays in to_concat are variable-shaped and/or any
        of the tensor arrays have a different shape than the others, a variable-shaped
        tensor array will be returned.
        """
        to_concat_types = [arr.type for arr in to_concat]
        if ArrowTensorType._need_variable_shaped_tensor_array(to_concat_types):
            return ArrowVariableShapedTensorArray.from_numpy([e for a in to_concat for e in a])
        else:
            storage = pa.concat_arrays([c.storage for c in to_concat])
            return ArrowTensorArray.from_storage(to_concat[0].type, storage)

    @classmethod
    def _chunk_tensor_arrays(cls, arrs: Sequence[Union['ArrowTensorArray', 'ArrowVariableShapedTensorArray']]) -> pa.ChunkedArray:
        """
        Create a ChunkedArray from multiple tensor arrays.
        """
        arrs_types = [arr.type for arr in arrs]
        if ArrowTensorType._need_variable_shaped_tensor_array(arrs_types):
            new_arrs = []
            for a in arrs:
                if isinstance(a.type, ArrowTensorType):
                    a = a.to_variable_shaped_tensor_array()
                assert isinstance(a.type, ArrowVariableShapedTensorType)
                new_arrs.append(a)
            arrs = new_arrs
        return pa.chunked_array(arrs)

    def to_variable_shaped_tensor_array(self) -> 'ArrowVariableShapedTensorArray':
        """
        Convert this tensor array to a variable-shaped tensor array.

        This is primarily used when concatenating multiple chunked tensor arrays where
        at least one chunked array is already variable-shaped and/or the shapes of the
        chunked arrays differ, in which case the resulting concatenated tensor array
        will need to be in the variable-shaped representation.
        """
        return ArrowVariableShapedTensorArray.from_numpy(self.to_numpy())