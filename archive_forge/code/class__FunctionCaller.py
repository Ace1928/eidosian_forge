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
class _FunctionCaller(_Node):
    """A base class for Convertibles that reference functions."""

    def __init__(self, node, function, enclosing_graph, first_function_input, type_attribute, function_attributes):
        """Initializes a _FunctionCaller.

    Args:
      node: As in _Node.
      function: As in _Node.
      enclosing_graph: As in _Node.
      first_function_input: The index of the first NodeDef input that is tied to
        the function inputs. It is assumed that the rest of the NodeDef inputs
        map one to one to function inputs.
      type_attribute: The name of the NodeDef attribute that defines the input
        types. It is assumed that the types listed here map one-to-one with the
        function inputs (that is, they do _not_ specify types for inputs that
        are not passed to functions).
      function_attributes: The names of the NodeDef attributes containing
        references to functions.
    """
        super(_FunctionCaller, self).__init__(node, function, enclosing_graph)
        self._first_function_input = first_function_input
        self._type_attribute = type_attribute
        self._function_attributes = function_attributes

    def converted_self(self):
        if self._converted_self is None:
            node = super(_FunctionCaller, self).converted_self().node
            converted_names = self._enclosing_graph.converted_function_names
            for attr_name in self._function_attributes:
                attr = node.attr[attr_name]
                if attr.HasField('func') and self._enclosing_graph.is_converted_function(attr.func.name):
                    attr.func.name = converted_names[attr.func.name]
                elif attr.HasField('list'):
                    for func in attr.list.func:
                        if self._enclosing_graph.is_converted_function(func.name):
                            func.name = converted_names[func.name]
        return self._converted_self

    def convert_variable_to_constant(self, incoming_edge, tensor_data):
        index = incoming_edge.destination.index
        for edge in self.outgoing_edges:
            dest = edge.destination.convertible
            if edge.source.index == index and isinstance(dest, _Function):
                dest.convert_variable_to_constant(edge, tensor_data)
        node = self.converted_self()
        if index >= self._first_function_input:
            node.update_dtype(self._type_attribute, index - self._first_function_input, tensor_data.dtype)

    def create_edges(self):
        """Creates edges related to a function caller.

    Edges from a function caller to its called functions are always edges from
    _inputs_ to _inputs_: a FunctionDef input is given by the caller, based on
    its own inputs.
    """
        super(_FunctionCaller, self).create_edges()
        for attr_name in self._function_attributes:
            attr = self._node.attr[attr_name]
            if attr.HasField('func'):
                function = self._enclosing_graph.functions[attr.func.name]
                for index in range(len(self._node.input) - self._first_function_input):
                    self.add_outgoing_edge(_Edge(_EndPoint(self, index + self._first_function_input), _EndPoint(function, index)))
            elif attr.HasField('list'):
                for func in attr.list.func:
                    function = self._enclosing_graph.functions[func.name]
                    for index in range(len(self._node.input) - self._first_function_input):
                        self.add_outgoing_edge(_Edge(_EndPoint(self, index + self._first_function_input), _EndPoint(function, index)))