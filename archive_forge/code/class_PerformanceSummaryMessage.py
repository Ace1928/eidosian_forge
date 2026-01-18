from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class PerformanceSummaryMessage(StatusMessage):
    """Message class to log PerformanceSummary parameters.

  This class acts as a relay between a multiprocess/multithread situation and
  the global status queue, from which the PerformanceSummary info gets consumed.
  """

    def __init__(self, message_time, uses_slice):
        """Creates a PerformanceSummaryMessage.

    Args:
      message_time: Float representing when message was created (seconds since
          Epoch).
      uses_slice: True if the command uses slice parallelism.
    """
        super(PerformanceSummaryMessage, self).__init__(message_time, process_id=None, thread_id=None)
        self.uses_slice = uses_slice

    def __str__(self):
        """Returns a string with a valid constructor for this message."""
        return '%s(%s, %s)' % (self.__class__.__name__, self.time, self.uses_slice)