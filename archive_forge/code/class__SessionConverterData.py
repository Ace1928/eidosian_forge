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
class _SessionConverterData(_ConverterData):
    """Container for Session-based conversion data."""

    def __init__(self, session, graph_def, output_node_names, variable_names_allowlist=None, variable_names_denylist=None):
        graph_def = graph_util.extract_sub_graph(graph_def, output_node_names)
        super(_SessionConverterData, self).__init__(graph_def, variable_names_allowlist=variable_names_allowlist, variable_names_denylist=variable_names_denylist)
        nodes_to_convert = []
        tensor_names_to_convert = []
        for node in self.graph_def.node:
            if node.op in ['Variable', 'VariableV2', 'VarHandleOp']:
                tensor_name = node.name
                if not self._should_convert(tensor_name):
                    continue
                if node.op == 'VarHandleOp':
                    tensor_name = tensor_name + '/Read/ReadVariableOp'
                nodes_to_convert.append(node)
                tensor_names_to_convert.append(tensor_name + ':0')
        if tensor_names_to_convert:
            converted_tensors = session.run(tensor_names_to_convert)
            for node, tensor_value in zip(nodes_to_convert, converted_tensors):
                self._tensor_data[node.name] = _TensorData(numpy=tensor_value, dtype=node.attr['dtype'].type, index=None)