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
class _RecoverableSession(_WrappedSession):
    """A wrapped session that recreates a session upon certain kinds of errors.

  The constructor is passed a SessionCreator object, not a session.

  Calls to `run()` are delegated to the wrapped session.  If a call raises the
  exception `tf.errors.AbortedError` or `tf.errors.UnavailableError`, the
  wrapped session is closed, and a new one is created by calling the factory
  again.
  """

    def __init__(self, sess_creator):
        """Create a new `_RecoverableSession`.

    The value returned by calling `sess_creator.create_session()` will be the
    session wrapped by this recoverable session.

    Args:
      sess_creator: A 'SessionCreator' to be wrapped by recoverable.
    """
        self._sess_creator = sess_creator
        _WrappedSession.__init__(self, self._create_session())

    def _create_session(self):
        while True:
            try:
                return self._sess_creator.create_session()
            except _PREEMPTION_ERRORS as e:
                logging.info('An error was raised while a session was being created. This may be due to a preemption of a connected worker or parameter server. A new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: %s', e)

    def _check_stop(self):
        try:
            if self._sess:
                return self._sess._check_stop()
            else:
                return True
        except _PREEMPTION_ERRORS as e:
            logging.info('An error was raised while considering whether the session is complete. This may be due to a preemption in a connected worker or parameter server. The current session will be closed and a new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: %s', e)
            self.close()
            self._sess = self._create_session()
            return False
        except Exception:
            return True

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        while True:
            try:
                if not self._sess:
                    self._sess = self._create_session()
                return self._sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
            except _PREEMPTION_ERRORS as e:
                logging.info('An error was raised. This may be due to a preemption in a connected worker or parameter server. The current session will be closed and a new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: %s', e)
                self.close()
                self._sess = None

    def run_step_fn(self, step_fn, raw_session, run_with_hooks):
        while True:
            try:
                if not self._sess:
                    self._sess = self._create_session()
                run_with_hooks = self._sess.run
                return self._sess.run_step_fn(step_fn, raw_session, run_with_hooks)
            except _PREEMPTION_ERRORS as e:
                logging.info('An error was raised. This may be due to a preemption in a connected worker or parameter server. The current session will be closed and a new session will be created. This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers. If this error occurs repeatedly, try increasing the number of parameter servers assigned to the job. Error: %s', e)
                self.close()
                self._sess = None