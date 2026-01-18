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
class OnSessionInitResponse:
    """Response from an on-session-init callback."""

    def __init__(self, action):
        """Constructor.

    Args:
      action: (`OnSessionInitAction`) Debugger action to take on session init.
    """
        _check_type(action, str)
        self.action = action