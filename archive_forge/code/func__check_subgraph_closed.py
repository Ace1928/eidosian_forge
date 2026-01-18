import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _check_subgraph_closed(n, reachable_by_input, input_nodes_set, name_to_input_name):
    """Checks to make sure node only connects to predecessor graph through inputs.

  Args:
    n: Node to check
    reachable_by_input: Nodes that are reachable by all inputs of subgraph
    input_nodes_set: The set of nodes that are "inputs".
    name_to_input_name: Maps from name to the list of inputs.

  Raises:
    TypeError: If the given node uses items past inputs directly.
  """
    next_to_visit = [n]
    visited = set()
    while next_to_visit:
        current_node = next_to_visit.pop()
        visited.add(current_node)
        if current_node in reachable_by_input and current_node not in input_nodes_set:
            raise TypeError('Node %s uses input %s not in input_nodes.' % (n, current_node))
        if current_node not in input_nodes_set:
            next_to_visit += [input_node for input_node in name_to_input_name[current_node] if input_node not in visited]