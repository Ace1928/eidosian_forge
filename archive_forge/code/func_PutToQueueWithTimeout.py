from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import errno
import logging
import multiprocessing
import threading
import traceback
from gslib.utils import constants
from gslib.utils import system_util
from six.moves import queue as Queue
def PutToQueueWithTimeout(queue, msg, timeout=STATUS_QUEUE_OP_TIMEOUT):
    """Puts an item to the status queue.

  If the queue is full, this function will timeout periodically and repeat
  until success. This avoids deadlock during shutdown by never making a fully
  blocking call to the queue, since Python signal handlers cannot execute
  in between instructions of the Python interpreter (see
  https://docs.python.org/2/library/signal.html for details).

  Args:
    queue: Queue class (typically the global status queue)
    msg: message to post to the queue.
    timeout: (optional) amount of time to wait before repeating put request.
  """
    put_success = False
    while not put_success:
        try:
            queue.put(msg, timeout=timeout)
            put_success = True
        except Queue.Full:
            pass