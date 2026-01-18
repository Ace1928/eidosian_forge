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
def _add_to_manifest(self, status_message):
    """Updates manifest file and pops file from tracking if needed."""
    if not self._manifest_manager:
        raise errors.Error('Received ManifestMessage but StatusTracker was not initialized with manifest path.')
    file_progress = self._tracked_file_progress.pop(status_message.source_url.url_string, None)
    self._manifest_manager.write_row(status_message, file_progress)