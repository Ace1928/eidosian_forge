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
@contextlib.contextmanager
def _task_queue_lock():
    """Context manager which acquires a lock when queue.get is unsafe.

  On Python 3.5 with spawn enabled, a race condition affects unpickling
  objects in queue.get calls. This manifests as an AttributeError intermittently
  thrown by ForkingPickler.loads, e.g.:

  AttributeError: Can't get attribute 'FileDownloadTask' on <module
  'googlecloudsdk.command_lib.storage.tasks.cp.file_download_task' from
  'googlecloudsdk/command_lib/storage/tasks/cp/file_download_task.py'

  Adding a lock around queue.get calls using this context manager resolves the
  issue.

  Yields:
    None, but acquires a lock which is released on exit.
  """
    get_is_unsafe = sys.version_info.major == 3 and sys.version_info.minor <= 5 and (multiprocessing_context.get_start_method() == 'spawn')
    try:
        if get_is_unsafe:
            _TASK_QUEUE_LOCK.acquire()
        yield
    finally:
        if get_is_unsafe:
            _TASK_QUEUE_LOCK.release()