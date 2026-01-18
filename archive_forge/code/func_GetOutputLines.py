from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import os.path
import subprocess
import threading
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
def GetOutputLines(cmd, timeout_sec, show_stderr=True, strip_output=False):
    """Run command and get its stdout as a list of lines.

  Args:
    cmd: List of executable and arg strings.
    timeout_sec: Command will be killed if it exceeds this.
    show_stderr: False to suppress stderr from the command.
    strip_output: Strip head/tail whitespace before splitting into lines.

  Returns:
    List of lines (without newlines).
  """
    stdout = _GetStdout(cmd, timeout_sec, show_stderr=show_stderr)
    if strip_output:
        stdout = stdout.strip()
    lines = stdout.splitlines()
    return lines