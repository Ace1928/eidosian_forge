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
def _default_global_step_tensor(self):
    """Returns the global_step from the default graph.

    Returns:
      The global step `Tensor` or `None`.
    """
    try:
        gs = ops.get_default_graph().get_tensor_by_name('global_step:0')
        if gs.dtype.base_dtype in [dtypes.int32, dtypes.int64]:
            return gs
        else:
            logging.warning("Found 'global_step' is not an int type: %s", gs.dtype)
            return None
    except KeyError:
        return None