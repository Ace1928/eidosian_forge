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
class _LiteSingleOperand(_LiteOperand):
    """A simple operand that is non-aggregated (i.e. most hints)."""

    def __init__(self, node):
        _LiteOperand.__init__(self)
        self.node = node
        self.name = _tensor_name_base(node.name)

    def flatten(self):
        return [self.name]

    def aggregate_and_return_name_for_input(self, out_graphdef):
        return self.name

    def aggregate_and_return_name_for_output(self, fused_op_name, index, out_graphdef):
        output_node = _copy.deepcopy(self.node)
        del output_node.input[:]
        output_node.input.append(_tensorflow_output_name(fused_op_name, index))
        out_graphdef.node.extend([output_node])
        return self.node.attr['type'].i

    def __str__(self):
        return str(self.name)