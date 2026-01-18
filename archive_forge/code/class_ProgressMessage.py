from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class ProgressMessage(StatusMessage):
    """Message class for a file/object/component progress.

  This class contains specific information about the current progress of a file,
  cloud object or single component.
  """

    def __init__(self, size, processed_bytes, src_url, message_time, dst_url=None, component_num=None, operation_name=None, process_id=None, thread_id=None):
        """Creates a ProgressMessage.

    Args:
      size: Integer for total size of this file/component, in bytes.
      processed_bytes: Integer for number of bytes already processed from that
          specific component, which means processed_bytes <= size.
      src_url: FileUrl/CloudUrl representing the source file.
      message_time: Float representing when message was created (seconds since
          Epoch).
      dst_url: FileUrl/CloudUrl representing the destination file, or None
          for unary operations like hashing.
      component_num: Indicates the component number, if any.
      operation_name: Name of the operation that is being made over that
          component.
      process_id: Process ID that produced this message (overridable for
          testing).
      thread_id: Thread ID that produced this message (overridable for testing).
    """
        super(ProgressMessage, self).__init__(message_time)
        self.size = size
        self.processed_bytes = processed_bytes
        self.component_num = component_num
        self.src_url = src_url
        self.dst_url = dst_url
        self.finished = size == processed_bytes
        self.operation_name = operation_name

    def __str__(self):
        """Returns a string with a valid constructor for this message."""
        dst_url_string = "'%s'" % self.dst_url if self.dst_url else None
        operation_name_string = "'%s'" % self.operation_name if self.operation_name else None
        return "%s(%s, %s, '%s', %s, dst_url=%s, component_num=%s, operation_name=%s, process_id=%s, thread_id=%s)" % (self.__class__.__name__, self.size, self.processed_bytes, self.src_url, self.time, dst_url_string, self.component_num, operation_name_string, self.process_id, self.thread_id)