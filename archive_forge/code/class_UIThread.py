from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import deque
import sys
import threading
import time
from six.moves import queue as Queue
from gslib.metrics import LogPerformanceSummaryParams
from gslib.metrics import LogRetryableError
from gslib.thread_message import FileMessage
from gslib.thread_message import FinalMessage
from gslib.thread_message import MetadataMessage
from gslib.thread_message import PerformanceSummaryMessage
from gslib.thread_message import ProducerThreadMessage
from gslib.thread_message import ProgressMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.thread_message import SeekAheadMessage
from gslib.thread_message import StatusMessage
from gslib.utils import parallelism_framework_util
from gslib.utils.unit_util import DecimalShort
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import PrettyTime
class UIThread(threading.Thread):
    """Responsible for centralized printing across multiple processes/threads.

  This class pulls status messages that are posted to the centralized status
  queue and coordinates displaying status and progress to the user. It is
  used only during calls to _ParallelApply, which in turn is called only when
  multiple threads and/or processes are used.

  This class sends the messages it receives to UIController, which
  decides the correct course of action.
  """

    def __init__(self, status_queue, stream, ui_controller, timeout=1):
        """Instantiates a _UIThread.

    Args:
      status_queue: Queue for reporting status updates.
      stream: Stream for printing messages.
      ui_controller: UI controller to manage messages.
      timeout: Timeout for getting a message.
    """
        super(UIThread, self).__init__()
        self.status_queue = status_queue
        self.stream = stream
        self.timeout = timeout
        self.ui_controller = ui_controller
        self.start()

    def run(self):
        try:
            while True:
                try:
                    status_message = self.status_queue.get(timeout=self.timeout)
                except Queue.Empty:
                    status_message = None
                    continue
                self.ui_controller.Call(status_message, self.stream)
                if status_message == _ZERO_TASKS_TO_DO_ARGUMENT:
                    break
        except Exception as e:
            self.stream.write('Exception in UIThread: %s\n' % e)