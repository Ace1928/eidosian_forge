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