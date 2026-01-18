from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import time
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.command_lib.deployment_manager import dm_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
import six
Wait for an operation to complete.

  Polls the operation requested approximately every second, showing a
  progress indicator. Returns when the operation has completed.

  Args:
    client: The API client to use.
    messages: The API message to use.
    operation_name: The name of the operation to wait on, as returned by
        operations.list.
    operation_description: A short description of the operation to wait on,
        such as 'create' or 'delete'. Will be displayed to the user.
    project: The name of the project that this operation belongs to.
    timeout: Number of seconds to wait for. Defaults to 3 minutes.

  Returns:
    The operation when it is done.

  Raises:
      HttpException: A http error response was received while executing api
          request. Will be raised if the operation cannot be found.
      OperationError: The operation finished with error(s).
      Error: The operation the timeout without completing.
  