from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class SeekAheadMessage(StatusMessage):
    """Message class for results obtained by SeekAheadThread().

  It estimates the number of objects and total size in case the task_queue
  cannot hold all tasks at once (only used in large operations).
  This class contains information about all the objects yet to be processed.
  """

    def __init__(self, num_objects, size, message_time):
        """Creates a SeekAheadMessage.

    Args:
      num_objects: Number of total objects that the SeekAheadThread estimates.
      size: Total size corresponding to the sum of the size of objects iterated
          by SeekAheadThread.
      message_time: Float representing when message was created (seconds since
          Epoch).
    """
        super(SeekAheadMessage, self).__init__(message_time)
        self.num_objects = num_objects
        self.size = size

    def __str__(self):
        """Returns a string with a valid constructor for this message."""
        return '%s(%s, %s, %s, process_id=%s, thread_id=%s)' % (self.__class__.__name__, self.num_objects, self.size, self.time, self.process_id, self.thread_id)