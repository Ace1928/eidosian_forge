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
def _create_serialize_to_tensor_saveable(self, saveable_factories):
    """Creates a saveable using the _serialize_to_tensor method."""
    suffix = saveable_compat.get_saveable_name(self.trackable) or ''
    saveable_name = _extract_saveable_name(self.object_proto.attributes[0].checkpoint_key) + suffix
    if not context.executing_eagerly():
        existing_op = self._checkpoint.restore_ops_by_name.get(saveable_name, None)
        if existing_op is not None:
            return ([existing_op], {})
        saveables_cache = self._checkpoint.saveables_cache.setdefault(self.trackable, {})
        if saveable_name in saveables_cache:
            return ([], {saveable_name: saveables_cache[saveable_name]})
    saveable = saveable_factories[trackable_utils.SERIALIZE_TO_TENSORS_NAME](name=saveable_name)
    if not context.executing_eagerly():
        saveables_cache[saveable_name] = saveable
    return ([], {saveable_name: saveable})