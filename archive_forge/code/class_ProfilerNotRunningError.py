import datetime
import os
import threading
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
class ProfilerNotRunningError(Exception):
    pass