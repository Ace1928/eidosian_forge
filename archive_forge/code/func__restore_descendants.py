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
def _restore_descendants(self, reader=None):
    """Restore the bound Trackable and dependencies (may be deferred)."""
    visit_queue = collections.deque([(self, self.trackable)])
    restore_ops = []
    tensor_saveables = {}
    python_positions = []
    registered_savers = collections.defaultdict(dict)
    while visit_queue:
        current_position, _ = visit_queue.popleft()
        new_restore_ops, new_tensor_saveables, new_python_positions, new_registered_savers = current_position._single_restore()
        restore_ops.extend(new_restore_ops)
        tensor_saveables.update(new_tensor_saveables)
        python_positions.extend(new_python_positions)
        for saver_name, trackable_map in new_registered_savers.items():
            registered_savers[saver_name].update(trackable_map)
        _queue_children_for_restoration(current_position, visit_queue)
        _queue_slot_variables(current_position, visit_queue)
    restore_ops.extend(current_position.checkpoint.restore_saveables(tensor_saveables, python_positions, registered_savers, reader=reader))
    return restore_ops