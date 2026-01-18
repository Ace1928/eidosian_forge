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
@deprecated('2020-07-01', 'use `tf.profiler.experimental.Profile` instead.')
class Profiler(object):
    """Context-manager eager profiler api.

  Example usage:
  ```python
  with Profiler("/path/to/logdir"):
    # do some work
  ```
  """

    def __init__(self, logdir):
        self._logdir = logdir

    def __enter__(self):
        start()

    def __exit__(self, typ, value, tb):
        result = stop()
        save(self._logdir, result)