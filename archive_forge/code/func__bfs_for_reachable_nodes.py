import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _bfs_for_reachable_nodes(target_nodes, name_to_input_name):
    """Breadth first search for reachable nodes from target nodes."""
    nodes_to_keep = set()
    next_to_visit = list(target_nodes)
    while next_to_visit:
        node = next_to_visit[0]
        del next_to_visit[0]
        if node in nodes_to_keep:
            continue
        nodes_to_keep.add(node)
        if node in name_to_input_name:
            next_to_visit += name_to_input_name[node]
    return nodes_to_keep