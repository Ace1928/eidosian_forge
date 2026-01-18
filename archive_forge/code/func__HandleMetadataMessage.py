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
def _HandleMetadataMessage(self, status_message):
    """Handles a MetadataMessage.

    Args:
      status_message: The MetadataMessage to be processed.
    """
    self.objects_finished += 1
    if self.num_objects_source >= EstimationSource.INDIVIDUAL_MESSAGES:
        self.num_objects_source = EstimationSource.INDIVIDUAL_MESSAGES
        self.num_objects += 1
    self.object_report_change = True
    self.last_progress_time = status_message.time
    if self.objects_finished == self.num_objects and self.num_objects_source == EstimationSource.PRODUCER_THREAD_FINAL:
        self.final_message = True