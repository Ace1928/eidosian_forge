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
@PublicAPI(stability='alpha')
class ArrowVariableShapedTensorArray(_ArrowTensorScalarIndexingMixin, pa.ExtensionArray):
    """
    An array of heterogeneous-shaped, homogeneous-typed tensors.

    This is the Arrow side of TensorArray for tensor elements that have differing
    shapes. Note that this extension only supports non-ragged tensor elements; i.e.,
    when considering each tensor element in isolation, they must have a well-defined
    shape. This extension also only supports tensor elements that all have the same
    number of dimensions.

    See Arrow docs for customizing extension arrays:
    https://arrow.apache.org/docs/python/extending_types.html#custom-extension-array-class
    """
    OFFSET_DTYPE = np.int32

    @classmethod
    def from_numpy(cls, arr: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]) -> 'ArrowVariableShapedTensorArray':
        """
        Convert an ndarray or an iterable of heterogeneous-shaped ndarrays to an array
        of heterogeneous-shaped, homogeneous-typed tensors.

        Args:
            arr: An ndarray or an iterable of heterogeneous-shaped ndarrays.

        Returns:
            An ArrowVariableShapedTensorArray containing len(arr) tensors of
            heterogeneous shape.
        """
        if not isinstance(arr, (list, tuple, np.ndarray)):
            raise ValueError(f'ArrowVariableShapedTensorArray can only be constructed from an ndarray or a list/tuple of ndarrays, but got: {type(arr)}')
        if len(arr) == 0:
            raise ValueError('Creating empty ragged tensor arrays is not supported.')
        shapes, sizes, raveled = ([], [], [])
        ndim = None
        for a in arr:
            a = np.asarray(a)
            if ndim is not None and a.ndim != ndim:
                raise ValueError(f'ArrowVariableShapedTensorArray only supports tensor elements that all have the same number of dimensions, but got tensor elements with dimensions: {ndim}, {a.ndim}')
            ndim = a.ndim
            shapes.append(a.shape)
            sizes.append(a.size)
            a = np.ravel(a, order='C')
            raveled.append(a)
        sizes = np.array(sizes)
        size_offsets = np.cumsum(sizes)
        total_size = size_offsets[-1]
        if all((_is_contiguous_view(curr, prev) for prev, curr in _pairwise(raveled))):
            np_data_buffer = raveled[-1].base
        else:
            np_data_buffer = np.concatenate(raveled)
        dtype = np_data_buffer.dtype
        pa_dtype = pa.from_numpy_dtype(dtype)
        if pa.types.is_string(pa_dtype):
            if dtype.byteorder == '>' or (dtype.byteorder == '=' and sys.byteorder == 'big'):
                raise ValueError(f'Only little-endian string tensors are supported, but got: {dtype}')
            pa_dtype = pa.binary(dtype.itemsize)
        if dtype.type is np.bool_:
            np_data_buffer = np.packbits(np_data_buffer, bitorder='little')
        data_buffer = pa.py_buffer(np_data_buffer)
        value_array = pa.Array.from_buffers(pa_dtype, total_size, [None, data_buffer])
        size_offsets = np.insert(size_offsets, 0, 0)
        offset_array = pa.array(size_offsets)
        data_array = pa.ListArray.from_arrays(offset_array, value_array)
        shape_array = pa.array(shapes)
        storage = pa.StructArray.from_arrays([data_array, shape_array], ['data', 'shape'])
        type_ = ArrowVariableShapedTensorType(pa_dtype, ndim)
        return pa.ExtensionArray.from_storage(type_, storage)

    def _to_numpy(self, index: Optional[int]=None, zero_copy_only: bool=False):
        """
        Helper for getting either an element of the array of tensors as an ndarray, or
        the entire array of tensors as a single ndarray.

        Args:
            index: The index of the tensor element that we wish to return as an
                ndarray. If not given, the entire array of tensors is returned as an
                ndarray.
            zero_copy_only: If True, an exception will be raised if the conversion to a
                NumPy array would require copying the underlying data (e.g. in presence
                of nulls, or for non-primitive types). This argument is currently
                ignored, so zero-copy isn't enforced even if this argument is true.

        Returns:
            The corresponding tensor element as an ndarray if an index was given, or
            the entire array of tensors as an ndarray otherwise.
        """
        if index is None:
            arrs = [self._to_numpy(i, zero_copy_only) for i in range(len(self))]
            return create_ragged_ndarray(arrs)
        data = self.storage.field('data')
        shapes = self.storage.field('shape')
        shape = shapes[index].as_py()
        value_type = data.type.value_type
        offset = data.offsets[index].as_py()
        data_buffer = data.buffers()[3]
        return _to_ndarray_helper(shape, value_type, offset, data_buffer)

    def to_numpy(self, zero_copy_only: bool=True):
        """
        Convert the entire array of tensors into a single ndarray.

        Args:
            zero_copy_only: If True, an exception will be raised if the conversion to a
                NumPy array would require copying the underlying data (e.g. in presence
                of nulls, or for non-primitive types). This argument is currently
                ignored, so zero-copy isn't enforced even if this argument is true.

        Returns:
            A single ndarray representing the entire array of tensors.
        """
        return self._to_numpy(zero_copy_only=zero_copy_only)