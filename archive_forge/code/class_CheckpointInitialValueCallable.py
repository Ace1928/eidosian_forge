import collections
import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.CheckpointInitialValueCallable', v1=[])
class CheckpointInitialValueCallable(object):
    """A callable object that returns a CheckpointInitialValue.

  See CheckpointInitialValue for more information.
  """

    def __init__(self, checkpoint_position):
        self._checkpoint_position = checkpoint_position

    @property
    def checkpoint_position(self):
        return self._checkpoint_position

    def __call__(self, shape=None, dtype=None, shard_info=None):
        return CheckpointInitialValue(self._checkpoint_position, shape, shard_info=shard_info)

    @property
    def restore_uid(self):
        return self._checkpoint_position.restore_uid