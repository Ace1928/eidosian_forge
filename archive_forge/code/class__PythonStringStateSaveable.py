import functools
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
class _PythonStringStateSaveable(saveable_object.SaveableObject):
    """Saves Python state in a checkpoint."""

    def __init__(self, name, state_callback, restore_callback):
        """Configure saving.

    Args:
      name: The checkpoint key to write to.
      state_callback: A function taking no arguments which returns a string.
        This function is run every time a checkpoint is written.
      restore_callback: A function taking a Python string, used to restore
        state.
    """

        def _state_callback_wrapper():
            with ops.init_scope():
                return state_callback()
        self._state_callback = _state_callback_wrapper
        self._restore_callback = restore_callback
        with ops.device('/cpu:0'):
            self._save_string = constant_op.constant('', dtype=dtypes.string)
        spec = saveable_object.SaveSpec(self._save_string, '', name, dtype=dtypes.string)
        super(_PythonStringStateSaveable, self).__init__(self._save_string, [spec], name)

    def feed_dict_additions(self):
        """When running a graph, indicates fresh state to feed."""
        return {self._save_string: self._state_callback()}

    def freeze(self):
        """Create a frozen `SaveableObject` which saves the current state."""

        def _constant_state():
            return constant_op.constant(self._state_callback(), dtype=dtypes.string)
        return trackable.NoRestoreSaveable(tensor=_constant_state, dtype=dtypes.string, name=self.name, device='cpu:0')