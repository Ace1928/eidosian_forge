import os
import threading
import time
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.platform import gfile
Implementation of abstract method in superclass.

    See doc of `NonInteractiveDebugWrapperSession.prepare_run_debug_urls()`
    for details. This implementation creates a run-specific subdirectory under
    self._session_root and stores information regarding run `fetches` and
    `feed_dict.keys()` in the subdirectory.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`
      feed_dict: Same as the `feed_dict` argument to `Session.run()`

    Returns:
      debug_urls: (`str` or `list` of `str`) file:// debug URLs to be used in
        this `Session.run()` call.
    