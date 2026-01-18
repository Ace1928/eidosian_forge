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
class StepContext:
    """Control flow instrument for the `step_fn` from `run_step_fn()`.

       Users of `step_fn` may perform `run()` calls without running hooks
       by accessing the `session`.  A `run()` call with hooks may be performed
       using `run_with_hooks()`.  Computation flow can be interrupted using
       `request_stop()`.
    """

    def __init__(self, session, run_with_hooks_fn):
        """Initializes the `step_context` argument for a `step_fn` invocation.

      Args:
        session: An instance of `tf.compat.v1.Session`.
        run_with_hooks_fn: A function for running fetches and hooks.
      """
        self._session = session
        self._run_with_hooks_fn = run_with_hooks_fn

    @property
    def session(self):
        return self._session

    def run_with_hooks(self, *args, **kwargs):
        """Same as `MonitoredSession.run`. Accepts the same arguments."""
        return self._run_with_hooks_fn(*args, **kwargs)

    def request_stop(self):
        """Exit the training loop by causing `should_stop()` to return `True`.

         Causes `step_fn` to exit by raising an exception.

      Raises:
        StopIteration
      """
        raise StopIteration('step_fn has requested the iterations to stop.')