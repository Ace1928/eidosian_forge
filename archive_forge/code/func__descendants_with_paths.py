import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _descendants_with_paths(self):
    """Returns a dict of descendants by node_id and paths to node.

    The names returned by this private method are subject to change.
    """
    all_nodes_with_paths = {}
    to_visit = collections.deque([0])
    all_nodes_with_paths[0] = 'root'
    path = all_nodes_with_paths.get(0)
    while to_visit:
        node_id = to_visit.popleft()
        obj = self._object_graph_proto.nodes[node_id]
        for child in obj.children:
            if child.node_id == 0 or child.node_id in all_nodes_with_paths.keys():
                continue
            path = all_nodes_with_paths.get(node_id)
            if child.node_id not in all_nodes_with_paths.keys():
                to_visit.append(child.node_id)
            all_nodes_with_paths[child.node_id] = path + '.' + child.local_name
    return all_nodes_with_paths