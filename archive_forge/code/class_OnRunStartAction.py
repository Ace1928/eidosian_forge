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
class OnRunStartAction:
    """Enum-like values for possible action to take on start of a run() call."""
    DEBUG_RUN = 'debug_run'
    PROFILE_RUN = 'profile_run'
    NON_DEBUG_RUN = 'non_debug_run'