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
def ProcessStructuredOut(result_holder):
    """Default processing for structured stdout from a non-threaded subprocess.

  Attempts to parse result_holder.stdstdout into an OutputMessage and return
  a tuple of output messages and resource content.

  Args:
    result_holder:  OperationResult

  Returns:
    ([str], [JSON]), Tuple of output messages and resource content.
  Raises:
    StructuredOutputError if result_holder can not be processed.
  """
    if result_holder.stdout:
        all_msg = result_holder.stdout if yaml.list_like(result_holder.stdout) else result_holder.stdout.strip().split('\n')
        msgs = []
        resources = []
        for msg_rec in all_msg:
            msg = ReadStructuredOutput(msg_rec)
            msgs.append(msg.body)
            if msg.resource_body:
                resources.append(msg.resource_body)
        return (msgs, resources)
    return (None, None)