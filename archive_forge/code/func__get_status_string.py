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