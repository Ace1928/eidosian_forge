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
class SharedProcessContext:
    """Context manager used to collect and set global state."""

    def __init__(self):
        """Collects global state in the main process."""
        if multiprocessing_context.get_start_method() == 'fork':
            return
        self._environment_variables = execution_utils.GetToolEnv()
        self._creds_context_manager = creds_context_managers.CredentialProvidersManager()
        self._key_store = encryption_util._key_store
        self._invocation_id = transport.INVOCATION_ID

    def __enter__(self):
        """Sets global state in child processes."""
        if multiprocessing_context.get_start_method() == 'fork':
            return
        self._environment_context_manager = execution_utils.ReplaceEnv(**self._environment_variables)
        self._environment_context_manager.__enter__()
        self._creds_context_manager.__enter__()
        encryption_util._key_store = self._key_store
        transport.INVOCATION_ID = self._invocation_id
        log.SetUserOutputEnabled(None)
        log.SetVerbosity(None)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Cleans up global state in child processes."""
        if multiprocessing_context.get_start_method() == 'fork':
            return
        self._environment_context_manager.__exit__(exc_type, exc_value, exc_traceback)
        self._creds_context_manager.__exit__(exc_type, exc_value, exc_traceback)