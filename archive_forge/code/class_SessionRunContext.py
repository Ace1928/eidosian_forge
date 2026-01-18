import collections
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.SessionRunContext'])
class SessionRunContext:
    """Provides information about the `session.run()` call being made.

  Provides information about original request to `Session.Run()` function.
  SessionRunHook objects can stop the loop by calling `request_stop()` of
  `run_context`. In the future we may use this object to add more information
  about run without changing the Hook API.
  """

    def __init__(self, original_args, session):
        """Initializes SessionRunContext."""
        self._original_args = original_args
        self._session = session
        self._stop_requested = False

    @property
    def original_args(self):
        """A `SessionRunArgs` object holding the original arguments of `run()`.

    If user called `MonitoredSession.run(fetches=a, feed_dict=b)`, then this
    field is equal to SessionRunArgs(a, b).

    Returns:
     A `SessionRunArgs` object
    """
        return self._original_args

    @property
    def session(self):
        """A TensorFlow session object which will execute the `run`."""
        return self._session

    @property
    def stop_requested(self):
        """Returns whether a stop is requested or not.

    If true, `MonitoredSession` stops iterations.
    Returns:
      A `bool`
    """
        return self._stop_requested

    def request_stop(self):
        """Sets stop requested field.

    Hooks can use this function to request stop of iterations.
    `MonitoredSession` checks whether this is called or not.
    """
        self._stop_requested = True