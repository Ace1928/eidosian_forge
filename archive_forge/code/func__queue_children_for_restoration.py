import collections
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import constants
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def _queue_children_for_restoration(checkpoint_position, visit_queue):
    """Queues the restoration of trackable's children or defers them."""
    trackable = checkpoint_position.trackable
    trackable_children = trackable._trackable_children()
    for child in checkpoint_position.object_proto.children:
        correspondence = checkpoint_position.checkpoint.object_by_proto_id.get(child.node_id, None)
        if correspondence is not None:
            continue
        child_position = checkpoint_position.create_child_position(child.node_id)
        local_object = trackable._lookup_dependency(child.local_name, trackable_children)
        child_proto = child_position.object_proto
        if local_object is None:
            if child_proto.HasField('has_checkpoint_values'):
                has_value = child_proto.has_checkpoint_values.value
            else:
                has_value = bool(child_proto.children or child_proto.attributes or child_proto.slot_variables or child_proto.HasField('registered_saver'))
            if has_value:
                trackable._deferred_dependencies.setdefault(child.local_name, []).append(child_position)
        elif child_position.bind_object(trackable=local_object):
            visit_queue.append((child_position, local_object))