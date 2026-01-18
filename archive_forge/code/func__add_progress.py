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