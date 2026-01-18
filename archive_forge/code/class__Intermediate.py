import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
class _Intermediate(_Node):
    """Specialization of _Node to intermediate ops."""

    def convert_variable_to_constant(self, incoming_edge, tensor_data):
        node = self.converted_self()
        node.update_dtype('T', incoming_edge.destination.index, tensor_data.dtype)
        if '_output_shapes' in node.node.attr:
            del node.node.attr['_output_shapes']
        for edge in self.outgoing_edges:
            edge.destination.convertible.convert_variable_to_constant(edge, tensor_data)