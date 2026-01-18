from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def _get_copy_nodes(self):
    """Find all Copy nodes in the loaded graph."""
    copy_nodes = []
    for node in self._node_inputs:
        if is_copy_node(node):
            copy_nodes.append(node)
    return copy_nodes