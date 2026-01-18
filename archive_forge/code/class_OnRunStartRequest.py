import abc
import re
import threading
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class OnRunStartRequest:
    """Request to an on-run-start callback.

  This callback is invoked during a run() call of the debug-wrapper
  session, immediately after the run() call counter is incremented.
  """

    def __init__(self, fetches, feed_dict, run_options, run_metadata, run_call_count, is_callable_runner=False):
        """Constructor of `OnRunStartRequest`.

    Args:
      fetches: Fetch targets of the run() call.
      feed_dict: The feed dictionary to the run() call.
      run_options: RunOptions input to the run() call.
      run_metadata: RunMetadata input to the run() call.
        The above four arguments are identical to the input arguments to the
        run() method of a non-wrapped TensorFlow session.
      run_call_count: 1-based count of how many run calls (including this one)
        has been invoked.
      is_callable_runner: (bool) whether a runner returned by
        Session.make_callable is being run.
    """
        self.fetches = fetches
        self.feed_dict = feed_dict
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.run_call_count = run_call_count
        self.is_callable_runner = is_callable_runner