import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
class _WrappedSession:
    """Wrapper around a `tf.compat.v1.Session`.

  This wrapper is used as a base class for various session wrappers
  that provide additional functionality such as monitoring, coordination,
  and recovery.

  In addition to the methods exported by `SessionInterface` the wrapper
  provides a method to check for stop and never raises exceptions from
  calls to `close()`.
  """

    def __init__(self, sess):
        """Creates a `_WrappedSession`.

    Args:
      sess: A `tf.compat.v1.Session` or `_WrappedSession` object.  The wrapped
        session.
    """
        self._sess = sess
        self._wrapped_is_stoppable = isinstance(self._sess, _WrappedSession)

    @property
    def graph(self):
        return self._sess.graph

    @property
    def sess_str(self):
        return self._sess.sess_str

    def should_stop(self):
        """Return true if this session should not be used anymore.

    Always return True if the session was closed.

    Returns:
      True if the session should stop, False otherwise.
    """
        if self._check_stop():
            return True
        if self._sess:
            return self._wrapped_is_stoppable and self._sess.should_stop()
        return True

    def _check_stop(self):
        """Hook for subclasses to provide their own stop condition.

    Returns:
      True if the session should stop, False otherwise.
    """
        return False

    def close(self):
        if self._sess:
            try:
                self._sess.close()
            except _PREEMPTION_ERRORS as e:
                logging.error('An error occurred when attempting to close the session. This may be due to a preemption in a connected worker or parameter server. Error: %s', e)
            finally:
                self._sess = None

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)

    def run_step_fn(self, step_fn, raw_session, run_with_hooks):
        run_with_hooks = run_with_hooks or self.run
        return step_fn(_MonitoredSession.StepContext(raw_session, run_with_hooks))