import abc
import typing
import warnings
import typing_extensions
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _convert_anonymous_fields(value, for_spec=False):
    """Type-checks and converts `value` for inclusion in an AnonymousExtensionType."""
    if isinstance(value, (int, float, bool, str, bytes, type(None), dtypes.DType, tensor_shape.TensorShape)):
        return value
    if isinstance(value, tuple):
        return tuple((_convert_anonymous_fields(v, for_spec) for v in value))
    if isinstance(value, typing.Mapping):
        return immutable_dict.ImmutableDict([(_convert_anonymous_fields(k, for_spec), _convert_anonymous_fields(v, for_spec)) for k, v in value.items()])
    if isinstance(value, (tensor.Tensor, composite_tensor.CompositeTensor)) and (not for_spec):
        return value
    if isinstance(value, type_spec.TypeSpec) and for_spec:
        return value
    raise ValueError(f'Cannot convert anonymous fields from an unsupported `value` argument: {value!r}.')