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
class OnRunEndRequest:
    """Request to an on-run-end callback.

  The callback is invoked immediately before the wrapped run() call ends.
  """

    def __init__(self, performed_action, run_metadata=None, client_graph_def=None, tf_error=None):
        """Constructor for `OnRunEndRequest`.

    Args:
      performed_action: (`OnRunStartAction`) Actually-performed action by the
        debug-wrapper session.
      run_metadata: run_metadata output from the run() call (if any).
      client_graph_def: (GraphDef) GraphDef from the client side, i.e., from
        the python front end of TensorFlow. Can be obtained with
        session.graph.as_graph_def().
      tf_error: (errors.OpError subtypes) TensorFlow OpError that occurred
        during the run (if any).
    """
        _check_type(performed_action, str)
        self.performed_action = performed_action
        if run_metadata is not None:
            _check_type(run_metadata, config_pb2.RunMetadata)
        self.run_metadata = run_metadata
        self.client_graph_def = client_graph_def
        self.tf_error = tf_error