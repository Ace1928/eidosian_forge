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
class ArrowVariableShapedTensorType(pa.ExtensionType):
    """
    Arrow ExtensionType for an array of heterogeneous-shaped, homogeneous-typed
    tensors.

    This is the Arrow side of TensorDtype for tensor elements with different shapes.
    Note that this extension only supports non-ragged tensor elements; i.e., when
    considering each tensor element in isolation, they must have a well-defined,
    non-ragged shape.

    See Arrow extension type docs:
    https://arrow.apache.org/docs/python/extending_types.html#defining-extension-types-user-defined-types
    """

    def __init__(self, dtype: pa.DataType, ndim: int):
        """
        Construct the Arrow extension type for array of heterogeneous-shaped tensors.

        Args:
            dtype: pyarrow dtype of tensor elements.
            ndim: The number of dimensions in the tensor elements.
        """
        self._ndim = ndim
        super().__init__(pa.struct([('data', pa.list_(dtype)), ('shape', pa.list_(pa.int64()))]), 'ray.data.arrow_variable_shaped_tensor')

    def to_pandas_dtype(self):
        """
        Convert Arrow extension type to corresponding Pandas dtype.

        Returns:
            An instance of pd.api.extensions.ExtensionDtype.
        """
        from ray.air.util.tensor_extensions.pandas import TensorDtype
        return TensorDtype((None,) * self.ndim, self.storage_type['data'].type.value_type.to_pandas_dtype())

    @property
    def ndim(self) -> int:
        """Return the number of dimensions in the tensor elements."""
        return self._ndim

    @property
    def scalar_type(self):
        """Returns the type of the underlying tensor elements."""
        data_field_index = self.storage_type.get_field_index('data')
        return self.storage_type[data_field_index].type.value_type

    def __reduce__(self):
        return (self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__()))

    def __arrow_ext_serialize__(self):
        return json.dumps(self._ndim).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        ndim = json.loads(serialized)
        dtype = storage_type['data'].type.value_type
        return cls(dtype, ndim)

    def __arrow_ext_class__(self):
        """
        ExtensionArray subclass with custom logic for this array of tensors
        type.

        Returns:
            A subclass of pd.api.extensions.ExtensionArray.
        """
        return ArrowVariableShapedTensorArray
    if _arrow_extension_scalars_are_subclassable():

        def __arrow_ext_scalar_class__(self):
            """
            ExtensionScalar subclass with custom logic for this array of tensors type.
            """
            return ArrowTensorScalar

    def __str__(self) -> str:
        dtype = self.storage_type['data'].type.value_type
        return f'numpy.ndarray(ndim={self.ndim}, dtype={dtype})'

    def __repr__(self) -> str:
        return str(self)
    if _arrow_supports_extension_scalars():

        def _extension_scalar_to_ndarray(self, scalar: pa.ExtensionScalar) -> np.ndarray:
            """
            Convert an ExtensionScalar to a tensor element.
            """
            data = scalar.value.get('data')
            raw_values = data.values
            shape = tuple(scalar.value.get('shape').as_py())
            value_type = raw_values.type
            offset = raw_values.offset
            data_buffer = raw_values.buffers()[1]
            return _to_ndarray_helper(shape, value_type, offset, data_buffer)