from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import threading
import time
from gslib import thread_message
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
class SeekAheadThread(threading.Thread):
    """Thread to estimate total work to be performed by all processes and threads.

  Because the ProducerThread can only buffer a certain number of tasks on the
  global task queue, it cannot reliably provide the total count or size of
  iterated results for operations involving many iterated arguments until it
  nears the end of iteration.

  This thread consumes an iterator that should be functionally identical
  to the ProducerThread, but iterates to the end without adding tasks to the
  global task queue in an effort to estimate the amount of total work that the
  call to Apply will perform. It should be used only for large operations, and
  thus it is created by the main ProducerThread only when the number of
  iterated arguments exceeds a threshold.

  This thread may produce an inaccurate estimate if its iterator produces
  different results than the iterator used by the ProducerThread. This can
  happen due to eventual listing consistency or due to the source being
  modified as iteration occurs.

  This thread estimates operations for top-level objects only;
  sub-operations (such as a parallel composite upload) should be reported via
  the iterator as a single object including the total number of bytes affected.
  """

    def __init__(self, seek_ahead_iterator, cancel_event, status_queue):
        """Initializes the seek ahead thread.

    Args:
      seek_ahead_iterator: Iterator matching the ProducerThread's args_iterator,
          but returning only object name and/or size in the result.
      cancel_event: threading.Event for signaling the
          seek-ahead iterator to terminate.
      status_queue: Status queue for posting summary of fully iterated results.
    """
        super(SeekAheadThread, self).__init__()
        self.status_queue = status_queue
        self.seek_ahead_iterator = seek_ahead_iterator
        self.cancel_event = cancel_event
        self.terminate = False
        self.start()

    def run(self):
        num_objects = 0
        num_data_bytes = 0
        try:
            for seek_ahead_result in self.seek_ahead_iterator:
                if self.terminate:
                    return
                if num_objects % constants.NUM_OBJECTS_PER_LIST_PAGE == 0:
                    if self.cancel_event.isSet():
                        return
                num_objects += seek_ahead_result.est_num_ops
                num_data_bytes += seek_ahead_result.data_bytes
        except OSError as e:
            return
        if self.cancel_event.isSet():
            return
        _PutToQueueWithTimeout(self.status_queue, thread_message.SeekAheadMessage(num_objects, num_data_bytes, time.time()))