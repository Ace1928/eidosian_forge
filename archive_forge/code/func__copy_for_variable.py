import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
def _copy_for_variable(self, original_var):
    """Create a new instance for the input trackable.

    Args:
      original_var: Input Variable object to be copied.
    """
    op_device = pydev.DeviceSpec.from_string(original_var.device).replace(device_type='CPU', device_index=0).to_string()
    with ops.device(op_device):
        new_var = UninitializedVariable(trainable=original_var.trainable, shape=original_var.shape, dtype=original_var.dtype, name=original_var._shared_name)
    self._object_map[original_var] = new_var