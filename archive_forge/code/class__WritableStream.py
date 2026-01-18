from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import upload_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class _WritableStream:
    """A write-only stream class that writes to the buffer queue."""

    def __init__(self, buffer_queue, buffer_condition, shutdown_event):
        """Initializes WritableStream.

    Args:
      buffer_queue (collections.deque): A queue where the data gets written.
      buffer_condition (threading.Condition): The condition object to wait on if
        the buffer is full.
      shutdown_event (threading.Event): Used for signaling the thread to
        terminate.
    """
        self._buffer_queue = buffer_queue
        self._buffer_condition = buffer_condition
        self._shutdown_event = shutdown_event

    def write(self, data):
        """Writes data to the buffer queue.

    This method writes the data in chunks of QUEUE_ITEM_MAX_SIZE. In most cases,
    the read operation is performed with size=QUEUE_ITEM_MAX_SIZE.
    Splitting the data in QUEUE_ITEM_MAX_SIZE chunks improves the performance.

    This method will be blocked if MAX_BUFFER_QUEUE_SIZE is reached to avoid
    writing all the data in-memory.

    Args:
      data (bytes): The bytes that should be added to the queue.

    Raises:
      _AbruptShutdownError: If self._shudown_event was set.
    """
        start = 0
        end = min(start + _QUEUE_ITEM_MAX_SIZE, len(data))
        while start < len(data):
            with self._buffer_condition:
                while len(self._buffer_queue) >= _MAX_BUFFER_QUEUE_SIZE and (not self._shutdown_event.is_set()):
                    self._buffer_condition.wait()
                if self._shutdown_event.is_set():
                    raise _AbruptShutdownError()
                self._buffer_queue.append(data[start:end])
                start = end
                end = min(start + _QUEUE_ITEM_MAX_SIZE, len(data))
                self._buffer_condition.notify_all()