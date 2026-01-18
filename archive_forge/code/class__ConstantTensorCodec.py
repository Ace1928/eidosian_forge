import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
class _ConstantTensorCodec:
    """Codec for Tensor."""

    def can_encode(self, pyobj):
        return isinstance(pyobj, tensor_lib.Tensor)

    def do_encode(self, tensor_value, encode_fn):
        """Returns an encoded `TensorProto` for the given `tf.Tensor`."""
        del encode_fn
        encoded_tensor = struct_pb2.StructuredValue()
        if isinstance(tensor_value, ops.EagerTensor):
            encoded_tensor.tensor_value.CopyFrom(tensor_util.make_tensor_proto(tensor_value.numpy()))
        elif tensor_value.op.type == 'Const':
            encoded_tensor.tensor_value.CopyFrom(tensor_value.op.get_attr('value'))
        else:
            raise nested_structure_coder.NotEncodableError(f'No encoder for object {str(tensor_value)} of type {type(tensor_value)}.')
        return encoded_tensor

    def can_decode(self, value):
        return value.HasField('tensor_value')

    def do_decode(self, value, decode_fn):
        """Returns the `tf.Tensor` encoded by the proto `value`."""
        del decode_fn
        tensor_proto = value.tensor_value
        tensor = constant(tensor_util.MakeNdarray(tensor_proto))
        return tensor