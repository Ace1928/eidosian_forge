from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.util import object_identity
def add_checkpoint_values_check(object_graph_proto):
    """Determines which objects have checkpoint values and save this to the proto.

  Args:
    object_graph_proto: A `TrackableObjectGraph` proto.
  """
    parents = {}
    checkpointed_trackables = object_identity.ObjectIdentitySet()
    checkpointed_trackables = set()
    for node_id, object_proto in enumerate(object_graph_proto.nodes):
        if object_proto.attributes or object_proto.slot_variables or object_proto.HasField('registered_saver'):
            checkpointed_trackables.add(node_id)
        for child_proto in object_proto.children:
            child = child_proto.node_id
            if child not in parents:
                parents[child] = set()
            parents[child].add(node_id)
    to_visit = set()
    to_visit.update(checkpointed_trackables)
    while to_visit:
        trackable = to_visit.pop()
        if trackable not in parents:
            continue
        current_parents = parents.pop(trackable)
        checkpointed_trackables.update(current_parents)
        for parent in current_parents:
            if parent in parents:
                to_visit.add(parent)
    for node_id, object_proto in enumerate(object_graph_proto.nodes):
        object_proto.has_checkpoint_values.value = bool(node_id in checkpointed_trackables)