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
def PrintFinalSummaryMessage(self, stream=sys.stderr):
    """Prints a final message to indicate operation succeeded.

    Args:
      stream: Stream to print messages. Usually sys.stderr, but customizable
              for testing.
    """
    string_to_print = 'Operation completed over %s objects' % DecimalShort(self.num_objects)
    if self.total_size:
        string_to_print += '/%s' % HumanReadableWithDecimalPlaces(self.total_size)
    remaining_width = self.console_width - len(string_to_print)
    if not self.quiet_mode:
        stream.write('\n' + string_to_print + '.' + max(remaining_width, 0) * ' ' + '\n')