from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
class UserRegisteredSpec(type_spec_module.TypeSpec):
    """TypeSpec to represent user-registered symbolic objects."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self._dtype = dtype
        self.dtype = dtype

    def _component_specs(self):
        raise NotImplementedError

    def _from_components(self, components):
        raise NotImplementedError

    def _serialize(self):
        raise NotImplementedError

    def _to_components(self, value):
        raise NotImplementedError

    def value_type(self):
        raise NotImplementedError