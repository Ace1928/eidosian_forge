import functools
import operator
from typing import Optional, Sequence, Type
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class _TensorShapeCodec:
    """Codec for `TensorShape`."""

    def can_encode(self, pyobj):
        return isinstance(pyobj, TensorShape)

    def do_encode(self, tensor_shape_value, encode_fn):
        del encode_fn
        encoded_tensor_shape = struct_pb2.StructuredValue()
        encoded_tensor_shape.tensor_shape_value.CopyFrom(tensor_shape_value.as_proto())
        return encoded_tensor_shape

    def can_decode(self, value):
        return value.HasField('tensor_shape_value')

    def do_decode(self, value, decode_fn):
        del decode_fn
        return TensorShape(value.tensor_shape_value)