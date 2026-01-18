from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
from gslib.thread_message import ProgressMessage
from gslib.utils import parallelism_framework_util
class ProgressCallbackWithTimeout(object):
    """Makes progress callbacks at least once every _TIMEOUT_SECONDS.

  This prevents wrong throughput calculation while not being excessively
  overwhelming.
  """

    def __init__(self, total_size, callback_func, start_bytes_per_callback=_START_BYTES_PER_CALLBACK, timeout=_TIMEOUT_SECONDS):
        """Initializes the callback with timeout.

    Args:
      total_size: Total bytes to process. If this is None, size is not known
          at the outset.
      callback_func: Func of (int: processed_so_far, int: total_bytes)
          used to make callbacks.
      start_bytes_per_callback: Lower bound of bytes per callback.
      timeout: Number maximum of seconds without a callback.

    """
        self._bytes_per_callback = start_bytes_per_callback
        self._callback_func = callback_func
        self._total_size = total_size
        self._last_time = time.time()
        self._timeout = timeout
        self._bytes_processed_since_callback = 0
        self._callbacks_made = 0
        self._total_bytes_processed = 0

    def Progress(self, bytes_processed):
        """Tracks byte processing progress, making a callback if necessary."""
        self._bytes_processed_since_callback += bytes_processed
        cur_time = time.time()
        if self._bytes_processed_since_callback > self._bytes_per_callback or (self._total_size is not None and self._total_bytes_processed + self._bytes_processed_since_callback >= self._total_size) or self._last_time - cur_time > self._timeout:
            self._total_bytes_processed += self._bytes_processed_since_callback
            if self._total_size is not None:
                bytes_sent = min(self._total_bytes_processed, self._total_size)
            else:
                bytes_sent = self._total_bytes_processed
            self._callback_func(bytes_sent, self._total_size)
            self._bytes_processed_since_callback = 0
            self._callbacks_made += 1
            self._last_time = cur_time