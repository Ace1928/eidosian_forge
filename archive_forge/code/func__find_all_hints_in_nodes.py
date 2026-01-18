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
def _find_all_hints_in_nodes(nodes):
    """Look at the all the input nodes and return a list of LiteFuncCall objs.

  Args:
    nodes: A TensorFlow graph_def to look for LiteFuncCalls.

  Returns:
    a list of `LifeFuncCall` objects in the form

  """
    func_calls = _collections.defaultdict(_LiteFuncCall)
    for node in nodes:
        attr = node.attr
        if OpHint.FUNCTION_UUID_ATTR not in attr or not attr[OpHint.FUNCTION_UUID_ATTR].s:
            continue
        uuid = attr[OpHint.FUNCTION_UUID_ATTR].s
        call_def = func_calls[uuid]
        call_def.uuid = uuid
        call_def.function_name = attr[OpHint.FUNCTION_NAME_ATTR].s
        call_def.level = attr[OpHint.FUNCTION_LEVEL_ATTR].i
        sort = attr[OpHint.FUNCTION_SORT_INDEX_ATTR].i if OpHint.FUNCTION_SORT_INDEX_ATTR in attr else None
        if sort == -1:
            sort = None
        aggregation = None
        if OpHint.FUNCTION_AGGREGATE_ATTR in attr:
            aggregation = _compat.as_text(attr[OpHint.FUNCTION_AGGREGATE_ATTR].s)
        if OpHint.CHILDREN_INPUTS_MAPPINGS in attr:
            call_def.children_inputs_mappings = _json.loads(_compat.as_text(attr[OpHint.CHILDREN_INPUTS_MAPPINGS].s))

        def put_operand(stuff, index, sort, operand, aggregation):
            """Add a given index into the function structure."""
            if sort is None:
                stuff[index] = _LiteSingleOperand(operand)
            else:
                if index not in stuff:
                    stuff[index] = _LiteAggregateOperand(aggregation)
                stuff[index].add(sort, operand)
        if OpHint.FUNCTION_INPUT_INDEX_ATTR in attr:
            put_operand(call_def.inputs, attr[OpHint.FUNCTION_INPUT_INDEX_ATTR].i, sort, node, aggregation)
        if OpHint.FUNCTION_OUTPUT_INDEX_ATTR in attr:
            put_operand(call_def.outputs, attr[OpHint.FUNCTION_OUTPUT_INDEX_ATTR].i, sort, node, aggregation)
        for a in attr:
            if a.startswith('_tflite_attr_'):
                call_def.params[a.replace('_tflite_attr_,', '')] = attr[a].tensor
    return func_calls