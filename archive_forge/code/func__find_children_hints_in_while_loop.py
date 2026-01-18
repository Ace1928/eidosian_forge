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
def _find_children_hints_in_while_loop(function_def, nodes_mapping):
    """Find children hints and all nodes inside the while loop.

  Args:
    function_def: Function def of the while loop.
    nodes_mapping: While loop input_arg : real node name.

  Returns:
    Ordered children hints and all re-mapped nodes inside the while loop.
  """
    new_nodes = []
    for node in function_def.node_def:
        for i, _ in enumerate(node.input):
            if node.input[i] in nodes_mapping:
                node.input[i] = nodes_mapping[node.input[i]]
        new_nodes.append(_copy.deepcopy(node))
    name_to_seq_num = _extract_topology_sequence_mapping(function_def.node_def)
    children_hints = _find_all_hints_in_nodes(new_nodes)
    children_hints_q = []
    for hint in children_hints.values():
        _, output_names = hint.flattened_inputs_and_outputs()
        seq = name_to_seq_num[output_names[0]]
        for output_name in output_names:
            seq = min(seq, name_to_seq_num[output_name])
        children_hints_q.append((seq, hint))
    children_hints_q.sort(key=lambda tup: tup[0])
    ordered_children_hints = [x[1] for x in children_hints_q]
    return (ordered_children_hints, new_nodes)