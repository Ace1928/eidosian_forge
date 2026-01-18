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
def _store_exception(target_function):
    """Decorator for storing exceptions raised from the thread targets.

  Args:
    target_function (function): Thread target to decorate.

  Returns:
    Decorator function.
  """

    @functools.wraps(target_function)
    def wrapper(self, *args, **kwargs):
        try:
            target_function(self, *args, **kwargs)
        except Exception as e:
            if not isinstance(self, TaskGraphExecutor):
                raise
            with self.thread_exception_lock:
                if self.thread_exception is None:
                    log.debug('Storing error to raise later: %s', e)
                    self.thread_exception = e
                else:
                    log.error(e)
                    log.debug(e, exc_info=sys.exc_info())
    return wrapper