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
class _Convertible(object):
    """An entity that can have variables converted to constants."""

    def __init__(self, enclosing_graph):
        self._enclosing_graph = enclosing_graph
        self._outgoing_edges = []
        self._converted_self = None

    def converted_self(self):
        """A copy of this Convertible to be modified during conversion.

    Returns:
      Implementations should return the copied instance, which in turn should
      be contained in converted_enclosing_graph(). This instance is the one that
      will be modified during conversion. Its main use will be in the
      implementations of convert_variable_to_constant().
    """
        raise NotImplementedError

    def convert_variable_to_constant(self, incoming_edge, tensor_data):
        """Converts a variable in this Convertible and its dependencies.

    This method should make sure that a converted copy of itself is present in
    the converted graph, and that all Convertibles depending on this one also go
    through the same process.

    Args:
      incoming_edge: The graph edge into this Convertible that is being
        converted to a constant.
      tensor_data: The tensor representing the constant.
    """
        raise NotImplementedError

    def create_edges(self):
        """Calls add_outgoing_edge for all edges known to this Convertible.

    This is used to build the graph dependencies, so that conversion of
    variables to constants can be properly propagated through the graph. Usually
    this method will call add_outgoing_edge() to all the Convertible inputs.
    """
        raise NotImplementedError

    def add_outgoing_edge(self, edge):
        """Adds an outgoing edge to the Convertible's list of edges.

    Args:
      edge: The outgoing edge (its source should be 'self').
    """
        self._outgoing_edges.append(edge)

    @property
    def converted_enclosing_graph(self):
        """The graph being converted."""
        return self._enclosing_graph.converted_self()

    @property
    def outgoing_edges(self):
        """The list of edges starting at this Convertible."""
        return self._outgoing_edges