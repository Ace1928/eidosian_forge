from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def ProcessStructuredErr(result_holder):
    """Default processing for structured stderr from non-threaded subprocess.

  Attempts to parse result_holder.stderr into an OutputMessage and return any
  status messages or raised errors.

  Args:
    result_holder:  OperationResult

  Returns:
    ([status messages], [errors]), Tuple of status messages and errors.
  Raises:
    StructuredOutputError if result_holder can not be processed.
  """
    if result_holder.stderr:
        all_msg = result_holder.stderr if yaml.list_like(result_holder.stderr) else result_holder.stderr.strip().split('\n')
        messages = []
        errors = []
        for msg_rec in all_msg:
            msg = ReadStructuredOutput(msg_rec)
            if msg.IsError():
                errors.append(msg.error_details.Format())
            else:
                messages.append(msg.body)
        return (messages, errors)
    return (None, None)