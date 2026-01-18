import contextlib
import os
import time
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _init_global_step(self, global_step=USE_DEFAULT):
    """Initializes global_step.

    Args:
      global_step: An integer Tensor of size 1 that counts steps. If set to
        USE_DEFAULT, creates global_step tensor.
    """
    if global_step is Supervisor.USE_DEFAULT:
        global_step = self._get_first_op_from_collection(ops.GraphKeys.GLOBAL_STEP)
        if global_step is None:
            global_step = self._default_global_step_tensor()
            if global_step is not None:
                ops.add_to_collection(ops.GraphKeys.GLOBAL_STEP, global_step)
    self._global_step = global_step