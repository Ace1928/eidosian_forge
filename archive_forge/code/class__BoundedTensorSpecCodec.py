from typing import Optional, Type
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
class _BoundedTensorSpecCodec:
    """Codec for `BoundedTensorSpec`."""

    def can_encode(self, pyobj):
        return isinstance(pyobj, BoundedTensorSpec)

    def do_encode(self, bounded_tensor_spec_value, encode_fn):
        """Returns an encoded proto for the given `tf.BoundedTensorSpec`."""
        encoded_bounded_tensor_spec = struct_pb2.StructuredValue()
        encoded_bounded_tensor_spec.bounded_tensor_spec_value.CopyFrom(struct_pb2.BoundedTensorSpecProto(shape=encode_fn(bounded_tensor_spec_value.shape).tensor_shape_value, dtype=encode_fn(bounded_tensor_spec_value.dtype).tensor_dtype_value, name=bounded_tensor_spec_value.name, minimum=tensor_util.make_tensor_proto(bounded_tensor_spec_value.minimum), maximum=tensor_util.make_tensor_proto(bounded_tensor_spec_value.maximum)))
        return encoded_bounded_tensor_spec

    def can_decode(self, value):
        return value.HasField('bounded_tensor_spec_value')

    def do_decode(self, value, decode_fn):
        btsv = value.bounded_tensor_spec_value
        name = btsv.name
        return BoundedTensorSpec(shape=decode_fn(struct_pb2.StructuredValue(tensor_shape_value=btsv.shape)), dtype=decode_fn(struct_pb2.StructuredValue(tensor_dtype_value=btsv.dtype)), minimum=tensor_util.MakeNdarray(btsv.minimum), maximum=tensor_util.MakeNdarray(btsv.maximum), name=name if name else None)