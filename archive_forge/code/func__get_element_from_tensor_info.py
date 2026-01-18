import weakref
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _get_element_from_tensor_info(tensor_info, graph):
    """Simplified copy of the deprecated `get_tensor_from_tensor_info`."""
    encoding = tensor_info.WhichOneof('encoding')
    if encoding == 'name':
        return graph.as_graph_element(tensor_info.name)
    elif encoding == 'coo_sparse':
        return sparse_tensor.SparseTensor(graph.get_tensor_by_name(tensor_info.coo_sparse.indices_tensor_name), graph.get_tensor_by_name(tensor_info.coo_sparse.values_tensor_name), graph.get_tensor_by_name(tensor_info.coo_sparse.dense_shape_tensor_name))
    elif encoding == 'composite_tensor':
        spec_proto = struct_pb2.StructuredValue(type_spec_value=tensor_info.composite_tensor.type_spec)
        spec = nested_structure_coder.decode_proto(spec_proto)
        components = [graph.get_tensor_by_name(component.name) for component in tensor_info.composite_tensor.components]
        return spec._from_components(components)
    else:
        raise ValueError(f"Invalid TensorInfo.encoding: {encoding}. Valid encodings are 'name', 'coo_sparse', and 'composite_tensor'.")