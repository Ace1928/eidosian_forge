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
def _LogStructuredStdOut(line):
    """Parse and log stdout text as an OutputMessage.

  Attempts to parse line into an OutputMessage and log any resource output or
  status messages accordingly. If message can not be parsed, raises a
  StructuredOutputError.

  Args:
    line: string, line of output read from stdout.

  Returns:
    Tuple: (str, object): Tuple of parsed OutputMessage body and
       processed resources or None.

  Raises: StructuredOutputError, if line can not be parsed.
  """
    msg = None
    resources = None
    if line:
        msg_rec = line.strip()
        msg = ReadStructuredOutput(msg_rec)
        if msg.resource_body:
            log.status.Print(msg.body)
            log.Print(msg.resource_body)
        else:
            log.Print(msg.body)
    return (msg.body, resources)