import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _model_ready_for_local_init(self, sess):
    """Checks if the model is ready to run local_init_op.

    Args:
      sess: A `Session`.

    Returns:
      A tuple (is_ready, msg), where is_ready is True if ready to run
      local_init_op and False otherwise, and msg is `None` if the model is
      ready to run local_init_op, a `String` with the reason why it is not ready
      otherwise.
    """
    return _ready(self._ready_for_local_init_op, sess, 'Model not ready for local init')