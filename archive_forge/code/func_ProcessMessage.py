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
def ProcessMessage(self, status_message, stream):
    """Processes a message from _MainThreadUIQueue or _UIThread.

    Args:
      status_message: The StatusMessage item to be processed.
      stream: Stream to print messages. Here only for SeekAheadThread
    """
    self.object_report_change = False
    if isinstance(status_message, ProducerThreadMessage):
        self._HandleProducerThreadMessage(status_message)
    elif isinstance(status_message, SeekAheadMessage):
        self._HandleSeekAheadMessage(status_message, stream)
    elif isinstance(status_message, FileMessage):
        if self._IsFile(status_message):
            self._HandleFileDescription(status_message)
        else:
            self._HandleComponentDescription(status_message)
        LogPerformanceSummaryParams(file_message=status_message)
    elif isinstance(status_message, ProgressMessage):
        self._HandleProgressMessage(status_message)
    elif isinstance(status_message, RetryableErrorMessage):
        LogRetryableError(status_message)
    elif isinstance(status_message, PerformanceSummaryMessage):
        self._HandlePerformanceSummaryMessage(status_message)
    self.old_progress.append(self._ThroughputInformation(self.new_progress, status_message.time))