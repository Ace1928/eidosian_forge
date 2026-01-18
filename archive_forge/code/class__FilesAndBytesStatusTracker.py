from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
import enum
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import metrics_util
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import scaled_integer
import six
class _FilesAndBytesStatusTracker(_StatusTracker, metrics_util.MetricsReporter):
    """See super class. Tracks both file count and byte amount."""

    def __init__(self, manifest_path=None):
        super(_FilesAndBytesStatusTracker, self).__init__()
        self._completed_files = 0
        self._processed_bytes = 0
        self._total_files_estimation = 0
        self._total_bytes_estimation = 0
        self._first_operation_time = None
        self._last_operation_time = None
        self._total_processed_bytes = 0
        self._window_start_time = None
        self._window_processed_bytes = 0
        self._window_throughput = None
        self._tracked_file_progress = {}
        if manifest_path:
            self._manifest_manager = manifest_util.ManifestManager(manifest_path)
        else:
            self._manifest_manager = None

    def _get_status_string(self):
        """See super class."""
        scaled_processed_bytes = scaled_integer.FormatBinaryNumber(self._processed_bytes, decimal_places=1)
        if self._total_files_estimation:
            file_progress_string = '{}/{}'.format(self._completed_files, self._total_files_estimation)
        else:
            file_progress_string = self._completed_files
        if self._total_bytes_estimation:
            scaled_total_bytes_estimation = scaled_integer.FormatBinaryNumber(self._total_bytes_estimation, decimal_places=1)
            bytes_progress_string = '{}/{}'.format(scaled_processed_bytes, scaled_total_bytes_estimation)
        else:
            bytes_progress_string = scaled_processed_bytes
        if self._window_throughput:
            throughput_addendum_string = ' | ' + self._window_throughput
        else:
            throughput_addendum_string = ''
        return 'Completed files {} | {}{}\r'.format(file_progress_string, bytes_progress_string, throughput_addendum_string)

    def _update_throughput(self, status_message, processed_bytes):
        """Updates stats and recalculates throughput if past threshold."""
        if self._first_operation_time is None:
            self._first_operation_time = status_message.time
            self._window_start_time = status_message.time
        else:
            self._last_operation_time = status_message.time
        self._window_processed_bytes += processed_bytes
        time_delta = status_message.time - self._window_start_time
        if time_delta > _THROUGHPUT_WINDOW_THRESHOLD_SECONDS:
            self._window_throughput = _get_formatted_throughput(self._window_processed_bytes, time_delta)
            self._window_start_time = status_message.time
            self._window_processed_bytes = 0

    def _add_to_workload_estimation(self, status_message):
        """Adds WorloadEstimatorMessage info to total workload estimation."""
        self._total_files_estimation += status_message.item_count
        self._total_bytes_estimation += status_message.size

    def _add_progress(self, status_message):
        """Track progress of a multipart file operation."""
        file_url_string = status_message.source_url.url_string
        if file_url_string not in self._tracked_file_progress:
            if status_message.total_components:
                self._tracked_file_progress[file_url_string] = FileProgress(component_count=status_message.total_components)
            else:
                self._tracked_file_progress[file_url_string] = FileProgress(component_count=1)
            if self._manifest_manager:
                self._tracked_file_progress[file_url_string].start_time = datetime.datetime.fromtimestamp(status_message.time, datetime.timezone.utc)
                self._tracked_file_progress[file_url_string].total_bytes_copied = 0
        component_tracker = self._tracked_file_progress[file_url_string].component_progress
        if status_message.component_number:
            component_number = status_message.component_number
        else:
            component_number = 0
        processed_component_bytes = status_message.current_byte - status_message.offset
        newly_processed_bytes = processed_component_bytes - component_tracker.get(component_number, 0)
        self._processed_bytes += newly_processed_bytes
        self._update_throughput(status_message, newly_processed_bytes)
        if self._manifest_manager:
            self._tracked_file_progress[file_url_string].total_bytes_copied += newly_processed_bytes
        if processed_component_bytes == status_message.length:
            component_tracker.pop(component_number, None)
            if not component_tracker:
                self._completed_files += 1
                if not self._manifest_manager:
                    del self._tracked_file_progress[file_url_string]
        else:
            component_tracker[component_number] = processed_component_bytes

    def _add_to_manifest(self, status_message):
        """Updates manifest file and pops file from tracking if needed."""
        if not self._manifest_manager:
            raise errors.Error('Received ManifestMessage but StatusTracker was not initialized with manifest path.')
        file_progress = self._tracked_file_progress.pop(status_message.source_url.url_string, None)
        self._manifest_manager.write_row(status_message, file_progress)

    def add_message(self, status_message):
        """See super class."""
        if isinstance(status_message, thread_messages.WorkloadEstimatorMessage):
            self._add_to_workload_estimation(status_message)
        elif isinstance(status_message, thread_messages.DetailedProgressMessage):
            self._set_source_and_destination_schemes(status_message)
            self._add_progress(status_message)
        elif isinstance(status_message, thread_messages.IncrementProgressMessage):
            self._completed_files += 1
        elif isinstance(status_message, thread_messages.ManifestMessage):
            self._add_to_manifest(status_message)

    def stop(self, exc_type, exc_val, exc_tb):
        super(_FilesAndBytesStatusTracker, self).stop(exc_type, exc_val, exc_tb)
        if self._first_operation_time is not None and self._last_operation_time is not None and (self._first_operation_time != self._last_operation_time):
            time_delta = self._last_operation_time - self._first_operation_time
            log.status.Print('\nAverage throughput: {}'.format(_get_formatted_throughput(self._processed_bytes, time_delta)))
            self._report_metrics(self._processed_bytes, time_delta, self._completed_files)