from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import auth
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberuncli
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
def FailureResult(self, out):
    """Processes the result of a failed kuberun command execution.

    Args:
      out: str, the output of the kuberun command

    Raises:
      An exception using 'out' as the message or a default message if empty
    """
    raise exceptions.Error(out if out else 'Command execution failed')