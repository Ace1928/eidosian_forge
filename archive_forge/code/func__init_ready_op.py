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
def _init_ready_op(self, ready_op=USE_DEFAULT, ready_for_local_init_op=USE_DEFAULT):
    """Initializes ready_op.

    Args:
      ready_op: `Tensor` to check if the model is initialized. If it's set to
        USE_DEFAULT, creates an op that checks all the variables are
        initialized.
      ready_for_local_init_op: `Tensor` to check if the model is ready to run
        local_init_op. If it's set to USE_DEFAULT, creates an op that checks all
        the global variables are initialized.
    """
    if ready_op is Supervisor.USE_DEFAULT:
        ready_op = self._get_first_op_from_collection(ops.GraphKeys.READY_OP)
        if ready_op is None:
            ready_op = variables.report_uninitialized_variables()
            ops.add_to_collection(ops.GraphKeys.READY_OP, ready_op)
    self._ready_op = ready_op
    if ready_for_local_init_op is Supervisor.USE_DEFAULT:
        ready_for_local_init_op = self._get_first_op_from_collection(ops.GraphKeys.READY_FOR_LOCAL_INIT_OP)
    self._ready_for_local_init_op = ready_for_local_init_op