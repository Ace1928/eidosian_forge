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
class SVTimerCheckpointThread(coordinator.LooperThread):
    """A thread to checkpoint on a timer."""

    def __init__(self, sv, sess):
        """Create a `SVTimerCheckpointThread`.

    Args:
      sv: A `Supervisor`.
      sess: A `Session`.
    """
        super(SVTimerCheckpointThread, self).__init__(sv.coord, sv.save_model_secs)
        self._sv = sv
        self._sess = sess

    def run_loop(self):
        logging.info('Saving checkpoint to path %s', self._sv.save_path)
        self._sv.saver.save(self._sess, self._sv.save_path, global_step=self._sv.global_step)
        if self._sv.summary_writer and self._sv.global_step is not None:
            current_step = training_util.global_step(self._sess, self._sv.global_step)
            self._sv.summary_writer.add_session_log(SessionLog(status=SessionLog.CHECKPOINT, checkpoint_path=self._sv.save_path), current_step)