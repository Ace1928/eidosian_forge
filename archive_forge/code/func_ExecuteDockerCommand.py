from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import re
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.core import log
def ExecuteDockerCommand(command):
    """Executes Docker CLI commands in subprocess.

  Just calls local_util.ExecuteCommand(cmd,...) and raises error for non-zero
  exit code.

  Args:
    command: (List[str]) Strings to send in as the command.

  Raises:
    ValueError: The input command is not a docker command.
    DockerError: An error occurred when executing the given docker command.
  """
    command_str = ' '.join(command)
    if not command_str.startswith('docker'):
        raise ValueError('`{}` is not a Docker command'.format('docker'))
    log.info('Running command: {}'.format(command_str))
    return_code = local_util.ExecuteCommand(command)
    if return_code != 0:
        error_msg = '\n        Docker failed with error code {code}.\n        Command: {cmd}\n        '.format(code=return_code, cmd=command_str)
        raise errors.DockerError(error_msg, command, return_code)