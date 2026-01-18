from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import subprocess
from gslib import exception
def ExecuteExternalCommand(command_and_flags):
    """Runs external terminal command.

  Args:
    command_and_flags (List[str]): Ordered command and flag strings.

  Returns:
    (stdout (str|None), stderr (str|None)) from running command.

  Raises:
    OSError for any issues running the command.
  """
    command_process = subprocess.Popen(command_and_flags, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command_stdout, command_stderr = command_process.communicate()
    if command_stdout is not None and (not isinstance(command_stdout, str)):
        command_stdout = command_stdout.decode()
    if command_stderr is not None and (not isinstance(command_stderr, str)):
        command_stderr = command_stderr.decode()
    if command_process.returncode != 0:
        raise exception.ExternalBinaryError(command_stderr)
    return (command_stdout, command_stderr)