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
class _CoordinatedSession(_WrappedSession):
    """A wrapped session that works with a `tf.Coordinator`.

  Calls to `run()` are delegated to the wrapped session.  If a call
  raises an exception, the exception is reported to the coordinator.

  In addition, after each call to `run()` this session ask the coordinator if
  the session should stop.  In that case it will join all the threads
  registered with the coordinator before returning.

  If the coordinator was requested to stop with an exception, that exception
  will be re-raised from the call to `run()`.
  """

    def __init__(self, sess, coord, stop_grace_period_secs=120):
        """Create a new `_CoordinatedSession`.

    Args:
      sess: A `tf.compat.v1.Session` object.  The wrapped session.
      coord: A `tf.train.Coordinator` object.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.
    """
        _WrappedSession.__init__(self, sess)
        self._coord = coord
        self._stop_grace_period_secs = stop_grace_period_secs

    def _check_stop(self):
        self._coord.raise_requested_exception()
        return self._coord.should_stop()

    def close(self):
        self._coord.request_stop()
        try:
            self._coord.join(stop_grace_period_secs=self._stop_grace_period_secs, ignore_live_threads=True)
        finally:
            try:
                _WrappedSession.close(self)
            except Exception:
                pass

    def run(self, *args, **kwargs):
        try:
            return self._sess.run(*args, **kwargs)
        except _PREEMPTION_ERRORS:
            raise
        except Exception as original_exception:
            try:
                self._coord.raise_requested_exception()
            except _PREEMPTION_ERRORS:
                raise
            except Exception:
                raise original_exception from None
            else:
                raise