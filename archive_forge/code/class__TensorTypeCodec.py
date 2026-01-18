import collections
import functools
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _TensorTypeCodec:
    """Codec for `TensorType`."""

    def can_encode(self, pyobj):
        return isinstance(pyobj, dtypes.DType)

    def do_encode(self, tensor_dtype_value, encode_fn):
        del encode_fn
        encoded_tensor_type = struct_pb2.StructuredValue()
        encoded_tensor_type.tensor_dtype_value = tensor_dtype_value.as_datatype_enum
        return encoded_tensor_type

    def can_decode(self, value):
        return value.HasField('tensor_dtype_value')

    def do_decode(self, value, decode_fn):
        del decode_fn
        return dtypes.DType(value.tensor_dtype_value)