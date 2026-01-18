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
class StatusMessageManager(object):
    """General manager for common functions shared by data and metadata managers.

  This subclass has the responsibility of having a common constructor and the
  same handler for SeekAheadMessages and ProducerThreadMessages.
  """

    class _ThroughputInformation(object):
        """Class that contains all information needed for throughput calculation.

    This _ThroughputInformation is used to track progress and time at several
    points of our operation.
    """

        def __init__(self, progress, report_time):
            """Constructor of _ThroughputInformation.

      Args:
        progress: The current progress, in bytes/second or objects/second.
        report_time: Float representing when progress was reported (seconds
            since Epoch).
      """
            self.progress = progress
            self.time = report_time

    def __init__(self, update_message_period=1, update_spinner_period=0.6, sliding_throughput_period=5, first_throughput_latency=10, quiet_mode=False, custom_time=None, verbose=False, console_width=80):
        """Instantiates a StatusMessageManager.

    Args:
      update_message_period: Minimum period for refreshing and  displaying
                             new information. A non-positive value will ignore
                             any time restrictions imposed by this field, but
                             it will affect throughput and time remaining
                             estimations.
      update_spinner_period: Minimum period for refreshing and displaying the
                             spinner. A non-positive value will ignore
                             any time restrictions imposed by this field.
      sliding_throughput_period: Sliding period for throughput calculation. A
                                 non-positive value will make it impossible to
                                 calculate the throughput.
      first_throughput_latency: Minimum waiting time before actually displaying
                                throughput info. A non-positive value will
                                ignore any time restrictions imposed by this
                                field.
      quiet_mode: If True, do not print status messages (but still process
                  them for analytics reporting as necessary).
      custom_time: If a custom start_time is desired. Used for testing.
      verbose: Tells whether or not the operation is on verbose mode.
      console_width: Width to display on console. This should not adjust the
                     visual output, just the space padding. For proper
                     visualization, we recommend setting this field to at least
                     80.
    """
        self.update_message_period = update_message_period
        self.update_spinner_period = update_spinner_period
        self.sliding_throughput_period = sliding_throughput_period
        self.first_throughput_latency = first_throughput_latency
        self.quiet_mode = quiet_mode
        self.custom_time = custom_time
        self.verbose = verbose
        self.console_width = console_width
        self.num_objects_source = EstimationSource.INDIVIDUAL_MESSAGES
        self.total_size_source = EstimationSource.INDIVIDUAL_MESSAGES
        self.num_objects = 0
        self.total_size = 0
        self.refresh_message_time = self.custom_time if self.custom_time else time.time()
        self.start_time = self.refresh_message_time
        self.refresh_spinner_time = self.refresh_message_time
        self.throughput = 0.0
        self.old_progress = deque()
        self.last_progress_time = 0
        self.spinner_char_list = ['/', '-', '\\', '|']
        self.current_spinner_index = 0
        self.objects_finished = 0
        self.num_objects = 0
        self.object_report_change = False
        self.final_message = False

    def GetSpinner(self):
        """Returns the current spinner character.

    Returns:
      char_to_print: Char to be printed as the spinner
    """
        return self.spinner_char_list[self.current_spinner_index]

    def UpdateSpinner(self):
        """Updates the current spinner character."""
        self.current_spinner_index = (self.current_spinner_index + 1) % len(self.spinner_char_list)

    def _HandleProducerThreadMessage(self, status_message):
        """Handles a ProducerThreadMessage.

    Args:
      status_message: The ProducerThreadMessage to be processed.
    """
        if status_message.finished:
            if self.num_objects_source >= EstimationSource.PRODUCER_THREAD_FINAL:
                self.num_objects_source = EstimationSource.PRODUCER_THREAD_FINAL
                self.num_objects = status_message.num_objects
            if self.total_size_source >= EstimationSource.PRODUCER_THREAD_FINAL and status_message.size:
                self.total_size_source = EstimationSource.PRODUCER_THREAD_FINAL
                self.total_size = status_message.size
            return
        if self.num_objects_source >= EstimationSource.PRODUCER_THREAD_ESTIMATE:
            self.num_objects_source = EstimationSource.PRODUCER_THREAD_ESTIMATE
            self.num_objects = status_message.num_objects
        if self.total_size_source >= EstimationSource.PRODUCER_THREAD_ESTIMATE and status_message.size:
            self.total_size_source = EstimationSource.PRODUCER_THREAD_ESTIMATE
            self.total_size = status_message.size

    def _HandleSeekAheadMessage(self, status_message, stream):
        """Handles a SeekAheadMessage.

    Args:
      status_message: The SeekAheadMessage to be processed.
      stream: Stream to print messages.
    """
        estimate_message = 'Estimated work for this command: objects: %s' % status_message.num_objects
        if status_message.size:
            estimate_message += ', total size: %s' % MakeHumanReadable(status_message.size)
            if self.total_size_source >= EstimationSource.SEEK_AHEAD_THREAD:
                self.total_size_source = EstimationSource.SEEK_AHEAD_THREAD
                self.total_size = status_message.size
        if self.num_objects_source >= EstimationSource.SEEK_AHEAD_THREAD:
            self.num_objects_source = EstimationSource.SEEK_AHEAD_THREAD
            self.num_objects = status_message.num_objects
        estimate_message += '\n'
        if not self.quiet_mode:
            stream.write(estimate_message)

    def _HandlePerformanceSummaryMessage(self, status_message):
        """Handles a PerformanceSummaryMessage.

    Args:
      status_message: The PerformanceSummaryMessage to be processed.
    """
        LogPerformanceSummaryParams(uses_slice=status_message.uses_slice)

    def ShouldTrackThroughput(self, cur_time):
        """Decides whether enough time has passed to start tracking throughput.

    Args:
      cur_time: current time.
    Returns:
      Whether or not we should track the throughput.
    """
        return cur_time - self.start_time >= self.first_throughput_latency

    def ShouldPrintProgress(self, cur_time):
        """Decides whether or not it is time for printing a new progress.

    Args:
      cur_time: current time.
    Returns:
      Whether or not we should print the progress.
    """
        sufficient_time_elapsed = cur_time - self.refresh_message_time >= self.update_message_period
        nonzero_report = self.num_objects
        return (sufficient_time_elapsed or self.object_report_change) and nonzero_report

    def ShouldPrintSpinner(self, cur_time):
        """Decides whether or not it is time for updating the spinner character.

    Args:
      cur_time: Current time.
    Returns:
      Whether or not we should update and print the spinner.
    """
        return cur_time - self.refresh_spinner_time > self.update_spinner_period and self.total_size

    def PrintSpinner(self, stream=sys.stderr):
        """Prints a spinner character.

    Args:
      stream: Stream to print messages. Usually sys.stderr, but customizable
              for testing.
    """
        self.UpdateSpinner()
        if not self.quiet_mode:
            stream.write(self.GetSpinner() + '\r')

    def UpdateThroughput(self, cur_time, cur_progress):
        """Updates throughput if the required period for calculation has passed.

    The throughput is calculated by taking all the progress (objects or bytes)
    processed within the last sliding_throughput_period seconds, and dividing
    that by the time period between the oldest progress time within that range
    and the last progress measurement, which are defined by oldest_progress[1]
    and last_progress_time, respectively. Among the pros of this approach,
    a connection break or a sudden change in throughput is quickly noticeable.
    Furthermore, using the last throughput measurement rather than the current
    time allows us to have a better estimation of the actual throughput.

    Args:
      cur_time: Current time to check whether or not it is time for a new
                throughput measurement.
      cur_progress: The current progress, in number of objects finished or in
                    bytes.
    """
        while len(self.old_progress) > 1 and cur_time - self.old_progress[0].time > self.sliding_throughput_period:
            self.old_progress.popleft()
        if not self.old_progress:
            return
        oldest_progress = self.old_progress[0]
        if self.last_progress_time == oldest_progress.time:
            self.throughput = 0
            return
        self.throughput = (cur_progress - oldest_progress.progress) / (self.last_progress_time - oldest_progress.time)
        self.throughput = max(0, self.throughput)

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