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
def _init_saver(self, saver=USE_DEFAULT):
    """Initializes saver.

    Args:
      saver: A `Saver` object. If set to USE_DEFAULT, create one that saves all
        the variables.
    """
    if saver is Supervisor.USE_DEFAULT:
        saver = self._get_first_op_from_collection(ops.GraphKeys.SAVERS)
        if saver is None and variables.global_variables():
            saver = saver_mod.Saver()
            ops.add_to_collection(ops.GraphKeys.SAVERS, saver)
    self._saver = saver