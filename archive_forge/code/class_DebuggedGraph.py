import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
class DebuggedGraph:
    """Data object representing debugging information about a tf.Graph.

  Includes `FuncGraph`s.

  Properties:
    name: Name of the graph (if any). May be `None` for non-function graphs.
    graph_id: Debugger-generated ID for the graph.
    inner_graph_ids: A list of the debugger-generated IDs for the graphs
      enclosed by this graph.
    outer_graph_id: If this graph is nested within an outer graph, ID of the
      outer graph. If this is an outermost graph, `None`.
  """

    def __init__(self, name, graph_id, outer_graph_id=None):
        self._name = name
        self._graph_id = graph_id
        self._outer_graph_id = outer_graph_id
        self._inner_graph_ids = []
        self._op_by_name = dict()
        self._op_consumers = collections.defaultdict(list)

    def add_inner_graph_id(self, inner_graph_id):
        """Add the debugger-generated ID of a graph nested within this graph.

    Args:
      inner_graph_id: The debugger-generated ID of the nested inner graph.
    """
        assert isinstance(inner_graph_id, str)
        self._inner_graph_ids.append(inner_graph_id)

    def add_op(self, graph_op_creation_digest):
        """Add an op creation data object.

    Args:
      graph_op_creation_digest: A GraphOpCreationDigest data object describing
        the creation of an op inside this graph.
    """
        if graph_op_creation_digest.op_name in self._op_by_name:
            raise ValueError('Duplicate op name: %s (op type: %s)' % (graph_op_creation_digest.op_name, graph_op_creation_digest.op_type))
        self._op_by_name[graph_op_creation_digest.op_name] = graph_op_creation_digest

    def add_op_consumer(self, src_op_name, src_slot, dst_op_name, dst_slot):
        """Add a consuming op for this op.

    Args:
      src_op_name: Name of the op of which the output tensor is being consumed.
      src_slot: 0-based output slot of the op being consumed.
      dst_op_name: Name of the consuming op (e.g., "Conv2D_3/BiasAdd")
      dst_slot: 0-based input slot of the consuming op that receives the tensor
        from this op.
    """
        self._op_consumers[src_op_name].append((src_slot, dst_op_name, dst_slot))

    @property
    def name(self):
        return self._name

    @property
    def graph_id(self):
        return self._graph_id

    @property
    def outer_graph_id(self):
        return self._outer_graph_id

    @property
    def inner_graph_ids(self):
        return self._inner_graph_ids

    def get_tensor_id(self, op_name, output_slot):
        """Get the ID of a symbolic tensor in this graph."""
        return self._op_by_name[op_name].output_tensor_ids[output_slot]

    def get_op_creation_digest(self, op_name):
        """Get the GraphOpCreationDigest for a op in the graph."""
        return self._op_by_name[op_name]

    def get_op_consumers(self, src_op_name):
        """Get all the downstream consumers of this op.

    Only data (non-control) edges are tracked.

    Args:
      src_op_name: Name of the op providing the tensor being consumed.

    Returns:
      A list of (src_slot, dst_op_name, dst_slot) tuples. In each item of
      the list:
        src_slot: 0-based output slot of the op of which the output tensor
          is being consumed.
        dst_op_name: Name of the consuming op (e.g., "Conv2D_3/BiasAdd")
        dst_slot: 0-based input slot of the consuming op that receives
          the tensor from this op.
    """
        return self._op_consumers[src_op_name]

    def to_json(self):
        return {'name': self.name, 'graph_id': self.graph_id, 'outer_graph_id': self._outer_graph_id, 'inner_graph_ids': self._inner_graph_ids}