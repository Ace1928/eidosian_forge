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
class _MonitoredSession:
    """See `MonitoredSession` or `SingularMonitoredSession`."""

    def __init__(self, session_creator, hooks, should_recover, stop_grace_period_secs=120):
        """Sets up a Monitored or Hooked Session.

    Args:
      session_creator: A factory object to create session. Typically a
        `ChiefSessionCreator` or a `WorkerSessionCreator`.
      hooks: An iterable of `SessionRunHook' objects.
      should_recover: A bool. Indicates whether to recover from `AbortedError`
        and `UnavailableError` or not.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.
    """
        self._graph_was_finalized = ops.get_default_graph().finalized
        self._hooks = hooks or []
        for h in self._hooks:
            h.begin()
        worker_context = distribute_coordinator_context.get_current_worker_context()
        if not session_creator and worker_context:
            session_creator = worker_context.session_creator()
        self._coordinated_creator = self._CoordinatedSessionCreator(session_creator=session_creator or ChiefSessionCreator(), hooks=self._hooks, stop_grace_period_secs=stop_grace_period_secs)
        if should_recover:
            self._sess = _RecoverableSession(self._coordinated_creator)
        else:
            self._sess = self._coordinated_creator.create_session()

    @property
    def graph(self):
        """The graph that was launched in this session."""
        if self._tf_sess() is None:
            return None
        return self._tf_sess().graph

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """Run ops in the monitored session.

    This method is completely compatible with the `tf.Session.run()` method.

    Args:
      fetches: Same as `tf.Session.run()`.
      feed_dict: Same as `tf.Session.run()`.
      options: Same as `tf.Session.run()`.
      run_metadata: Same as `tf.Session.run()`.

    Returns:
      Same as `tf.Session.run()`.
    """
        return self._sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    def run_step_fn(self, step_fn):
        """Run ops using a step function.

    Args:
      step_fn: A function or a method with a single argument of type
        `StepContext`.  The function may use methods of the argument to perform
        computations with access to a raw session.  The returned value of the
        `step_fn` will be returned from `run_step_fn`, unless a stop is
        requested.  In that case, the next `should_stop` call will return True.
        Example usage:
            ```python
            with tf.Graph().as_default():
              c = tf.compat.v1.placeholder(dtypes.float32)
              v = tf.add(c, 4.0)
              w = tf.add(c, 0.5)
              def step_fn(step_context):
                a = step_context.session.run(fetches=v, feed_dict={c: 0.5})
                if a <= 4.5:
                  step_context.request_stop()
                  return step_context.run_with_hooks(fetches=w,
                                                     feed_dict={c: 0.1})

              with tf.MonitoredSession() as session:
                while not session.should_stop():
                  a = session.run_step_fn(step_fn)
            ```
            Hooks interact with the `run_with_hooks()` call inside the
                 `step_fn` as they do with a `MonitoredSession.run` call.

    Returns:
      Returns the returned value of `step_fn`.

    Raises:
      StopIteration: if `step_fn` has called `request_stop()`.  It may be
        caught by `with tf.MonitoredSession()` to close the session.
      ValueError: if `step_fn` doesn't have a single argument called
        `step_context`. It may also optionally have `self` for cases when it
        belongs to an object.
    """
        step_fn_arguments = function_utils.fn_args(step_fn)
        if step_fn_arguments != ('step_context',) and step_fn_arguments != ('self', 'step_context'):
            raise ValueError("`step_fn` may either have one `step_context` argument, or `self` and `step_context` arguments if it's an instance method. Got {} instead.".format(step_fn_arguments))
        return self._sess.run_step_fn(step_fn, self._tf_sess(), run_with_hooks=None)

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

    def should_stop(self):
        return self._sess is None or self._sess.should_stop()

    def close(self):
        self._close_internal()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type in [errors.OutOfRangeError, StopIteration]:
            exception_type = None
        self._close_internal(exception_type)
        return exception_type is None

    class _CoordinatedSessionCreator(SessionCreator):
        """Factory for _CoordinatedSession."""

        def __init__(self, session_creator, hooks, stop_grace_period_secs):
            self._session_creator = session_creator
            self._hooks = hooks
            self.coord = None
            self.tf_sess = None
            self._stop_grace_period_secs = stop_grace_period_secs

        def create_session(self):
            """Creates a coordinated session."""
            self.tf_sess = self._session_creator.create_session()
            self.coord = coordinator.Coordinator(clean_stop_exception_types=[])
            if ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
                queue_runner.start_queue_runners(sess=self.tf_sess, coord=self.coord)
            for hook in self._hooks:
                hook.after_create_session(self.tf_sess, self.coord)
            return _CoordinatedSession(_HookedSession(self.tf_sess, self._hooks), self.coord, self._stop_grace_period_secs)

    def _close_internal(self, exception_type=None):
        try:
            if not exception_type:
                for h in self._hooks:
                    h.end(self._coordinated_creator.tf_sess)
        finally:
            try:
                if self._sess is None:
                    raise RuntimeError('Session is already closed.')
                self._sess.close()
            finally:
                self._sess = None
                self._coordinated_creator.tf_sess = None
                self._coordinated_creator.coord = None
                if not self._graph_was_finalized:
                    ops.get_default_graph()._unsafe_unfinalize()

    def _is_closed(self):
        """Return True if the monitored session is closed.

    For tests only.

    Returns:
      A boolean.
    """
        return self._coordinated_creator.tf_sess is None

    def _tf_sess(self):
        """Return underlying tf.compat.v1.Session object.

    Warning: accessing the returned object in user code is likely to cause races
    or "flaky tests".

    Returns:
      A tf.compat.v1.Session object.
    """
        return self._coordinated_creator.tf_sess