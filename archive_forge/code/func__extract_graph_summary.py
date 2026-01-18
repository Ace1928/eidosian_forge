import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _extract_graph_summary(graph_def):
    """Extracts useful information from the graph and returns them."""
    name_to_input_name = {}
    name_to_node = {}
    name_to_seq_num = {}
    seq = 0
    for node in graph_def.node:
        n = _node_name(node.name)
        name_to_node[n] = node
        name_to_input_name[n] = [_node_name(x) for x in node.input]
        if '_class' in node.attr:
            for colocated_node_name in node.attr['_class'].list.s:
                name_to_input_name[n].append(_get_colocated_node_name(colocated_node_name))
        name_to_seq_num[n] = seq
        seq += 1
    return (name_to_input_name, name_to_node, name_to_seq_num)