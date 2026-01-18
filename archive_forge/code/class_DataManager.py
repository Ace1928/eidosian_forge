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
class DataManager(StatusMessageManager):
    """Manages shared state for data operations.

  This manager is specific for data operations. Among its main functions,
  it receives incoming StatusMessages, storing all necessary data
  about the current and past states of the system necessary to display to the
  UI. It also provides methods for calculating metrics such as throughput and
  estimated time remaining. Finally, it provides methods for displaying messages
  to the UI.
  """

    class _ProgressInformation(object):
        """Class that contains all progress information needed for a given file.

    This _ProgressInformation is used as the value associated with a file_name
    in the dict that stores the information about all processed files.
    """

        def __init__(self, size):
            """Constructor of _ProgressInformation.

      Args:
        size: The total size of the file.
      """
            self.new_progress_sum = 0
            self.existing_progress_sum = 0
            self.dict = {}
            self.size = size

    def __init__(self, update_message_period=1, update_spinner_period=0.6, sliding_throughput_period=5, first_throughput_latency=10, quiet_mode=False, custom_time=None, verbose=False, console_width=None):
        """Instantiates a DataManager.

    See argument documentation in StatusMessageManager base class.
    """
        super(DataManager, self).__init__(update_message_period=update_message_period, update_spinner_period=update_spinner_period, sliding_throughput_period=sliding_throughput_period, first_throughput_latency=first_throughput_latency, quiet_mode=quiet_mode, custom_time=custom_time, verbose=verbose, console_width=console_width)
        self.first_item = True
        self.total_progress = 0
        self.new_progress = 0
        self.existing_progress = 0
        self.individual_file_progress = {}
        self.component_total = 0
        self.finished_components = 0
        self.existing_components = 0

    def GetProgress(self):
        """Gets the progress for a DataManager.

    Returns:
      The number of processed bytes in this operation.
    """
        return self.new_progress

    def _HandleFileDescription(self, status_message):
        """Handles a FileMessage that describes a file.

    Args:
      status_message: the FileMessage to be processed.
    """
        if not status_message.finished:
            if self.first_item and (not self.custom_time):
                self.refresh_message_time = status_message.time
                self.start_time = self.refresh_message_time
                self.last_throughput_time = self.refresh_message_time
                self.first_item = False
            file_name = status_message.src_url.url_string
            status_message.size = status_message.size if status_message.size else 0
            self.individual_file_progress[file_name] = self._ProgressInformation(status_message.size)
            if self.num_objects_source >= EstimationSource.INDIVIDUAL_MESSAGES:
                self.num_objects_source = EstimationSource.INDIVIDUAL_MESSAGES
                self.num_objects += 1
            if self.total_size_source >= EstimationSource.INDIVIDUAL_MESSAGES:
                self.total_size_source = EstimationSource.INDIVIDUAL_MESSAGES
                self.total_size += status_message.size
            self.object_report_change = True
        else:
            self.objects_finished += 1
            file_name = status_message.src_url.url_string
            file_progress = self.individual_file_progress[file_name]
            total_bytes_transferred = file_progress.new_progress_sum + file_progress.existing_progress_sum
            self.total_progress += file_progress.size - total_bytes_transferred
            self.new_progress += file_progress.size - total_bytes_transferred
            self.last_progress_time = status_message.time
            del self.individual_file_progress[file_name]
            self.object_report_change = True
            if self.objects_finished == self.num_objects and self.num_objects_source == EstimationSource.PRODUCER_THREAD_FINAL:
                self.final_message = True

    def _IsFile(self, file_message):
        """Tells whether or not this FileMessage represent a file.

    This is needed because FileMessage is used by both files and components.

    Args:
      file_message: The FileMessage to be analyzed.
    Returns:
      Whether or not this represents a file.
    """
        message_type = file_message.message_type
        return message_type == FileMessage.FILE_DOWNLOAD or message_type == FileMessage.FILE_UPLOAD or message_type == FileMessage.FILE_CLOUD_COPY or (message_type == FileMessage.FILE_DAISY_COPY) or (message_type == FileMessage.FILE_LOCAL_COPY) or (message_type == FileMessage.FILE_REWRITE) or (message_type == FileMessage.FILE_HASH)

    def _HandleComponentDescription(self, status_message):
        """Handles a FileMessage that describes a component.

    Args:
      status_message: The FileMessage to be processed.
    """
        if status_message.message_type == FileMessage.EXISTING_COMPONENT and (not status_message.finished):
            self.existing_components += 1
            file_name = status_message.src_url.url_string
            file_progress = self.individual_file_progress[file_name]
            key = (status_message.component_num, status_message.dst_url)
            file_progress.dict[key] = (0, status_message.size)
            file_progress.existing_progress_sum += status_message.size
            self.total_progress += status_message.size
            self.existing_progress += status_message.size
        elif status_message.message_type == FileMessage.COMPONENT_TO_UPLOAD or status_message.message_type == FileMessage.COMPONENT_TO_DOWNLOAD:
            if not status_message.finished:
                self.component_total += 1
                if status_message.message_type == FileMessage.COMPONENT_TO_DOWNLOAD:
                    file_name = status_message.src_url.url_string
                    file_progress = self.individual_file_progress[file_name]
                    file_progress.existing_progress_sum += status_message.bytes_already_downloaded
                    key = (status_message.component_num, status_message.dst_url)
                    file_progress.dict[key] = (0, status_message.bytes_already_downloaded)
                    self.total_progress += status_message.bytes_already_downloaded
                    self.existing_progress += status_message.bytes_already_downloaded
            else:
                self.finished_components += 1
                file_name = status_message.src_url.url_string
                file_progress = self.individual_file_progress[file_name]
                key = (status_message.component_num, status_message.dst_url)
                last_update = file_progress.dict[key] if key in file_progress.dict else (0, 0)
                self.total_progress += status_message.size - sum(last_update)
                self.new_progress += status_message.size - sum(last_update)
                self.last_progress_time = status_message.time
                file_progress.new_progress_sum += status_message.size - sum(last_update)
                file_progress.dict[key] = (status_message.size - last_update[1], last_update[1])

    def _HandleProgressMessage(self, status_message):
        """Handles a ProgressMessage that tracks progress of a file or component.

    Args:
      status_message: The ProgressMessage to be processed.
    """
        file_name = status_message.src_url.url_string
        file_progress = self.individual_file_progress[file_name]
        key = (status_message.component_num, status_message.dst_url)
        last_update = file_progress.dict[key] if key in file_progress.dict else (0, 0)
        status_message.processed_bytes -= last_update[1]
        file_progress.new_progress_sum += status_message.processed_bytes - last_update[0]
        self.total_progress += status_message.processed_bytes - last_update[0]
        self.new_progress += status_message.processed_bytes - last_update[0]
        file_progress.dict[key] = (status_message.processed_bytes, last_update[1])
        self.last_progress_time = status_message.time

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

    def PrintProgress(self, stream=sys.stderr):
        """Prints progress and throughput/time estimation.

    If a ProducerThreadMessage or SeekAheadMessage has been provided,
    it outputs the number of files completed, number of total files,
    the current progress, the total size, and the percentage it
    represents.
    If none of those have been provided, it only includes the number of files
    completed, the current progress and total size (which might be updated),
    with no percentage as we do not know if more files are coming.
    It may also include time estimation (available only given
    ProducerThreadMessage or SeekAheadMessage provided) and throughput. For that
    to happen, there is an extra condition of at least first_throughput_latency
    seconds having been passed since the UIController started, and that
    either the ProducerThread or the SeekAheadThread have estimated total
    number of files and total size.

    Args:
      stream: Stream to print messages. Usually sys.stderr, but customizable
              for testing.
    """
        total_remaining = self.total_size - self.total_progress
        if self.throughput:
            time_remaining = total_remaining / self.throughput
        else:
            time_remaining = None
        char_to_print = self.GetSpinner()
        if self.num_objects_source <= EstimationSource.SEEK_AHEAD_THREAD:
            objects_completed = '[' + DecimalShort(self.objects_finished) + '/' + DecimalShort(self.num_objects) + ' files]'
        else:
            objects_completed = '[' + DecimalShort(self.objects_finished) + ' files]'
        bytes_progress = '[%s/%s]' % (BytesToFixedWidthString(self.total_progress), BytesToFixedWidthString(self.total_size))
        if self.total_size_source <= EstimationSource.SEEK_AHEAD_THREAD:
            if self.num_objects == self.objects_finished:
                percentage = '100'
            else:
                percentage = '%3d' % min(99, int(100 * float(self.total_progress) / self.total_size))
            percentage_completed = percentage + '% Done'
        else:
            percentage_completed = ''
        if self.refresh_message_time - self.start_time > self.first_throughput_latency:
            throughput = BytesToFixedWidthString(self.throughput) + '/s'
            if self.total_size_source <= EstimationSource.PRODUCER_THREAD_ESTIMATE and self.throughput:
                time_remaining_str = 'ETA ' + PrettyTime(time_remaining)
            else:
                time_remaining_str = ''
        else:
            throughput = ''
            time_remaining_str = ''
        format_str = '{char_to_print} {objects_completed}{bytes_progress} {percentage_completed} {throughput} {time_remaining_str}'
        string_to_print = format_str.format(char_to_print=char_to_print, objects_completed=objects_completed, bytes_progress=bytes_progress, percentage_completed=percentage_completed, throughput=throughput, time_remaining_str=time_remaining_str)
        remaining_width = self.console_width - len(string_to_print)
        if not self.quiet_mode:
            stream.write(string_to_print + max(remaining_width, 0) * ' ' + '\r')

    def CanHandleMessage(self, status_message):
        """Determines whether this manager is suitable for handling status_message.

    Args:
      status_message: The StatusMessage object to be analyzed.
    Returns:
      True if this message can be properly handled by this manager,
      False otherwise.
    """
        if isinstance(status_message, (SeekAheadMessage, ProducerThreadMessage, FileMessage, ProgressMessage, FinalMessage, RetryableErrorMessage, PerformanceSummaryMessage)):
            return True
        return False