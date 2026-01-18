from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.objects import bulk_restore_objects_task
from googlecloudsdk.command_lib.storage.tasks.objects import restore_object_task
from googlecloudsdk.core import log
def _raise_if_invalid_flag_combination(args, execution_mode, invalid_flags):
    """Raises error if invalid combination of flags found in user input.

  Args:
    args (parser_extensions.Namespace): User input object.
    execution_mode (ExecutionMode): Determined by presence of --async flag.
    invalid_flags (list[str]): Flags as `args` attributes.

  Raises:
    error.Error: Flag incompatible with execution mode.
  """
    for invalid_flag in invalid_flags:
        if getattr(args, invalid_flag):
            raise errors.Error('{} execution does not support flag: {}. See help text with --help.'.format(execution_mode.value, invalid_flag))