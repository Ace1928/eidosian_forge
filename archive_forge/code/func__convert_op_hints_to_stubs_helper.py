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
def _convert_op_hints_to_stubs_helper(graph_def, write_callback=lambda sess, graph_def: None):
    """Converts a graph_def to a new graph_def where all op hints are stubbed.

  Args:
    graph_def: A graph def that we should convert.
    write_callback: A function pointer that can be used to write intermediate
      steps of graph transformation (optional).

  Returns:
    A new stubbed graph_def.
  """
    hints = _find_all_hints_in_nodes(graph_def.node)
    hints_q = []
    for hint in hints.values():
        hints_q.append((hint.level, hint.uuid))
    hints_q.sort(key=lambda tup: tup[0])
    for i in range(len(hints_q) - 1, -1, -1):
        level, hint_uuid = hints_q[i]
    curr_graph_def = graph_def
    del graph_def
    for i in range(len(hints_q) - 1, -1, -1):
        level, hint_uuid = hints_q[i]
        if level >= 2:
            children_hints, curr_graph_def, function_def_nodes = _find_children_hints(hints[hint_uuid], curr_graph_def)
            assert len(children_hints) > 0
            children_inputs_mappings = hints[hint_uuid].children_inputs_mappings
            for j, child_hint in enumerate(children_hints):
                if j == 0:
                    for mapping in children_inputs_mappings['parent_first_child_input']:
                        parent_input_index = _get_correct_mapping(mapping['parent_ophint_input_index'], hints[hint_uuid].inputs)
                        child_input_index = _get_correct_mapping(mapping['first_child_ophint_input_index'], child_hint.inputs)
                        child_hint.inputs[child_input_index] = hints[hint_uuid].inputs[parent_input_index]
                else:
                    for mapping in children_inputs_mappings['internal_children_input_output']:
                        input_index = _get_correct_mapping(mapping['child_input_index'], child_hint.inputs)
                        output_index = _get_correct_mapping(mapping['child_output_index'], children_hints[j - 1].outputs)
                        child_hint.inputs[input_index] = children_hints[j - 1].outputs[output_index]
                if j == len(children_hints) - 1:
                    for mapping in children_inputs_mappings['parent_last_child_output']:
                        parent_output_index = _get_correct_mapping(mapping['parent_output_index'], hints[hint_uuid].outputs)
                        child_output_index = _get_correct_mapping(mapping['child_output_index'], child_hint.outputs)
                        child_hint.outputs[child_output_index] = hints[hint_uuid].outputs[parent_output_index]
            for j, child_hint in enumerate(children_hints):
                curr_graph_def = _convert_single_op_hint_to_stub(child_hint, curr_graph_def, function_def_nodes, j == len(children_hints) - 1)
        else:
            curr_graph_def = _convert_single_op_hint_to_stub(hints[hint_uuid], curr_graph_def)
            write_callback(curr_graph_def, 'initial')
    curr_graph_def = _remove_redundant_stack_unstack(curr_graph_def)
    return curr_graph_def