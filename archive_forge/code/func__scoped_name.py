from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.tensorflow_stub import dtypes
def _scoped_name(name_scope, node_name):
    """Returns scoped name for a node as a string in the form '<scope>/<node
    name>'.

    Args:
      name_scope: a string representing a scope name, similar to that of tf.name_scope.
      node_name: a string representing the current node name.

    Returns
      A string representing a scoped name.
    """
    if name_scope:
        return '%s/%s' % (name_scope, node_name)
    return node_name