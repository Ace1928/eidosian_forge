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
class _ReadVariable(_Node):
    """Specialization of _Node to ReadVariableOp."""

    def convert_variable_to_constant(self, incoming_edge, tensor_data):
        node = self.converted_self().node
        node.Clear()
        node.name = self._node.name
        node.op = 'Identity'
        node.input.append(self._node.input[0])
        node.attr['T'].CopyFrom(self._node.attr['dtype'])
        if '_class' in self._node.attr:
            node.attr['_class'].CopyFrom(self._node.attr['_class'])
        if self._function is not None:
            for edge in self.outgoing_edges:
                index = edge.destination.index
                dest = edge.destination.convertible.converted_self()
                if isinstance(dest, _Node):
                    input_name_parts = dest.node.input[index].split(':')
                    if len(input_name_parts) > 1 and input_name_parts[1] == 'value':
                        input_name_parts[1] = 'output'
                        dest.node.input[index] = ':'.join(input_name_parts)