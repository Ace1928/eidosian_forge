from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import functools
import multiprocessing
import sys
import threading
from googlecloudsdk.api_lib.storage.gcs_json import patch_apitools_messages
from googlecloudsdk.command_lib import crash_handling
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_buffer
from googlecloudsdk.command_lib.storage.tasks import task_graph as task_graph_module
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds_context_managers
from googlecloudsdk.core.util import platforms
from six.moves import queue
@_store_exception
def _add_executable_tasks_to_queue(self):
    """Sends executable tasks to consumer threads in child processes."""
    task_wrapper = None
    while True:
        if task_wrapper is None:
            task_wrapper = self._executable_tasks.get()
            if task_wrapper == _SHUTDOWN:
                break
        reached_process_limit = self._process_count >= self._max_process_count
        try:
            self._task_queue.put(task_wrapper, block=reached_process_limit)
            task_wrapper = None
        except queue.Full:
            if self._idle_thread_count.acquire(block=False):
                self._idle_thread_count.release()
            else:
                self._add_worker_process()