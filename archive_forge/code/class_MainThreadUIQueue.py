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
class MainThreadUIQueue(object):
    """Handles status display and processing in the main thread / master process.

  This class emulates a queue to cover main-thread activity before or after
  Apply, as well as for the single-threaded, single-process case, i.e.,
  _SequentialApply. When multiple threads or processes are used during calls
  to Apply, the main thread is waiting for work to complete, and this queue
  must remain unused until Apply returns. Code producing arguments for
  Apply (such as the NameExpansionIterator) must not post messages to this
  queue to avoid race conditions with the UIThread.

  This class sends the messages it receives to UIController, which
  decides the correct course of action.
  """

    def __init__(self, stream, ui_controller):
        """Instantiates a _MainThreadUIQueue.

    Args:
      stream: Stream for printing messages.
      ui_controller: UIController to manage messages.
    """
        super(MainThreadUIQueue, self).__init__()
        self.ui_controller = ui_controller
        self.stream = stream

    def put(self, status_message, timeout=None):
        self.ui_controller.Call(status_message, self.stream)