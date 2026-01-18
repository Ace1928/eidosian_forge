from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
from gslib.thread_message import ProgressMessage
from gslib.utils import parallelism_framework_util
class FileProgressCallbackHandler(object):
    """Tracks progress info for large operations like file copy or hash.

      Information is sent to the status_queue, which will print it in the
      UI Thread.
  """

    def __init__(self, status_queue, start_byte=0, override_total_size=None, src_url=None, component_num=None, dst_url=None, operation_name=None):
        """Initializes the callback handler.

    Args:
      status_queue: Queue for posting status messages for UI display.
      start_byte: The beginning of the file component, if one is being used.
      override_total_size: The size of the file component, if one is being used.
      src_url: FileUrl/CloudUrl representing the source file.
      component_num: Indicates the component number, if any.
      dst_url: FileUrl/CloudUrl representing the destination file, or None
        for unary operations like hashing.
      operation_name: String representing the operation name
    """
        self._status_queue = status_queue
        self._start_byte = start_byte
        self._override_total_size = override_total_size
        self._component_num = component_num
        self._src_url = src_url
        self._dst_url = dst_url
        self._operation_name = operation_name
        self._last_byte_written = False

    def call(self, last_byte_processed, total_size):
        """Gathers information describing the operation progress.

    Actual message is printed to stderr by UIThread.

    Args:
      last_byte_processed: The last byte processed in the file. For file
                           components, this number should be in the range
                           [start_byte:start_byte + override_total_size].
      total_size: Total size of the ongoing operation.
    """
        if self._last_byte_written:
            return
        if self._override_total_size:
            total_size = self._override_total_size
        parallelism_framework_util.PutToQueueWithTimeout(self._status_queue, ProgressMessage(total_size, last_byte_processed - self._start_byte, self._src_url, time.time(), component_num=self._component_num, operation_name=self._operation_name, dst_url=self._dst_url))
        if total_size and last_byte_processed - self._start_byte == total_size:
            self._last_byte_written = True