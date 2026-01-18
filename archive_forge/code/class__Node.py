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
class _Node(_Convertible):
    """A Convertible NodeDef."""

    def __init__(self, node, function, enclosing_graph):
        super(_Node, self).__init__(enclosing_graph)
        self._node = node
        self._function = function

    def __str__(self):
        return self._node.name

    @staticmethod
    def new(node, function, enclosing_graph):
        """Creates a new _Node base on its operation type."""
        if node.op in ['VariableV2', 'VarHandleOp', 'Placeholder']:
            return _VarHandle(node, function, enclosing_graph)
        elif node.op == 'Case':
            return _Case(node, function, enclosing_graph)
        elif node.op == 'Merge':
            return _Merge(node, function, enclosing_graph)
        elif node.op == 'PartitionedCall':
            return _PartitionedCall(node, function, enclosing_graph)
        elif node.op == 'StatefulPartitionedCall':
            return _PartitionedCall(node, function, enclosing_graph)
        elif node.op == 'ReadVariableOp':
            return _ReadVariable(node, function, enclosing_graph)
        elif node.op == 'ResourceGather':
            return _ResourceGather(node, function, enclosing_graph)
        elif node.op == 'ResourceGatherNd':
            return _ResourceGatherNd(node, function, enclosing_graph)
        elif node.op in ['If', 'StatelessIf']:
            return _If(node, function, enclosing_graph)
        elif node.op in ['While', 'StatelessWhile']:
            return _While(node, function, enclosing_graph)
        elif node.op in ['Enter', 'Exit', 'Identity', 'NextIteration', 'Switch', '_SwitchN']:
            return _Intermediate(node, function, enclosing_graph)
        else:
            return _Node(node, function, enclosing_graph)

    @property
    def node(self):
        return self._node

    @property
    def container(self):
        """The node container (either a graph or a function)."""
        if self._function is not None:
            return self._function.function
        return self._enclosing_graph.graph_def

    def converted_self(self):
        """The NodeDef to be converted.

    Returns:
      The NodeDef to be converted, which can come from either a graph for a
      function. Derived classes should call this (via 'super') to make sure the
      node is retrieved from the right place.
    """
        if self._converted_self is None:
            source = self._function or self._enclosing_graph
            self._converted_self = source.converted_self().nodes[self._node.name]
        return self._converted_self

    def convert_variable_to_constant(self, incoming_edge, tensor_data):
        pass

    def create_edges(self):
        for index, name in enumerate(self._node.input):
            if name[0] == '^':
                continue
            source = self.resolve_input(name)
            source.convertible.add_outgoing_edge(_Edge(source, _EndPoint(self, index)))

    def resolve_input(self, input_name):
        """Resolves an input into its _EndPoint.

    A NodeDef's input name can refer to either global NodeDefs (in the
    GraphDef's node list), a NodeDef in a function's node list, or a Function
    (in the GraphDef's function library). The name can also carry semantic
    information, depending on whether it starts with "^". This method handles
    all that logic in order to find the object to which the input name refers
    to.

    Args:
      input_name: The input name to resolve.

    Returns:
      The object referred to by 'input_name'.
    """
        name_elts = input_name.split(':')
        source_name = name_elts[0]
        if source_name[0] == '^':
            source_name = source_name[1:]
        source_index = 0
        if len(name_elts) > 1 and name_elts[-1].isnumeric():
            source_index = int(name_elts[-1])
        if self._function is None:
            return _EndPoint(self._enclosing_graph.nodes[source_name], source_index)
        if source_index != 0 or source_name in self._function.nodes:
            return _EndPoint(self._function.nodes[source_name], source_index)
        inputs = [i.name for i in self._function.function.signature.input_arg]
        return _EndPoint(self._function, inputs.index(source_name))

    def update_dtype(self, attr_name, index, dtype):
        """Changes the type of a given input.

    Args:
      attr_name: The NodeDef attribute containing the type to change.
      index: The index of the input type to change.
      dtype: The type to change to.
    """
        attr = self._node.attr[attr_name]
        num_types = 0
        if attr.HasField('list'):
            types = attr.list.type
            num_types = len(types)
            if num_types > index:
                types[index] = dtype
                return
        elif attr.HasField('type'):
            num_types = 1
            if index == 0:
                attr.type = dtype
                return
        raise ValueError(f'`index` {index:d} is out of range for node({self._node.name}).attr({attr_name}), which has {num_types:d} elements.')