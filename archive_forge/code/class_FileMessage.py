from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class FileMessage(StatusMessage):
    """Marks start or end of an operation for a file, cloud object or component.

  This class contains general information about each file/component. With that,
  information such as total size and estimated time remaining may be calculated
  beforehand.
  """
    FILE_DOWNLOAD = 1
    FILE_UPLOAD = 2
    FILE_CLOUD_COPY = 3
    FILE_LOCAL_COPY = 4
    FILE_DAISY_COPY = 5
    FILE_REWRITE = 6
    FILE_HASH = 7
    COMPONENT_TO_UPLOAD = 8
    EXISTING_COMPONENT = 9
    COMPONENT_TO_DOWNLOAD = 10
    EXISTING_OBJECT_TO_DELETE = 11

    def __init__(self, src_url, dst_url, message_time, size=None, finished=False, component_num=None, message_type=None, bytes_already_downloaded=None, process_id=None, thread_id=None):
        """Creates a FileMessage.

    Args:
      src_url: FileUrl/CloudUrl representing the source file.
      dst_url: FileUrl/CloudUrl representing the destination file.
      message_time: Float representing when message was created (seconds since
          Epoch).
      size: Total size of this file/component, in bytes.
      finished: Boolean to indicate whether this is starting or finishing
          a file/component transfer.
      component_num: Component number, if dealing with a component.
      message_type: Type of the file/component.
      bytes_already_downloaded: Specific field for resuming downloads. When
          starting a component download, it tells how many bytes were already
          downloaded.
      process_id: Process ID that produced this message (overridable for
          testing).
      thread_id: Thread ID that produced this message (overridable for testing).
    """
        super(FileMessage, self).__init__(message_time, process_id=process_id, thread_id=thread_id)
        self.src_url = src_url
        self.dst_url = dst_url
        self.size = size
        self.component_num = component_num
        self.finished = finished
        self.message_type = message_type
        self.bytes_already_downloaded = bytes_already_downloaded

    def __str__(self):
        """Returns a string with a valid constructor for this message."""
        return "%s('%s', '%s', %s, size=%s, finished=%s, component_num=%s, message_type=%s, bytes_already_downloaded=%s, process_id=%s, thread_id=%s)" % (self.__class__.__name__, self.src_url, self.dst_url, self.time, self.size, self.finished, self.component_num, self.message_type, self.bytes_already_downloaded, self.process_id, self.thread_id)