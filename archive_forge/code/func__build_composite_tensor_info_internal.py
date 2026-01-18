from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _build_composite_tensor_info_internal(tensor):
    """Utility function to build TensorInfo proto from a CompositeTensor."""
    spec = tensor._type_spec
    tensor_info = meta_graph_pb2.TensorInfo()
    spec_proto = nested_structure_coder.encode_structure(spec)
    tensor_info.composite_tensor.type_spec.CopyFrom(spec_proto.type_spec_value)
    for component in nest.flatten(tensor, expand_composites=True):
        tensor_info.composite_tensor.components.add().CopyFrom(build_tensor_info_internal(component))
    return tensor_info