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
def StreamOutputJson(cmd, event_timeout_sec, show_stderr=True):
    """Run command and get its output streamed as an iterable of dicts.

  Args:
    cmd: List of executable and arg strings.
    event_timeout_sec: Command will be killed if we don't get a JSON line for
      this long. (This is not the same as timeout_sec above).
    show_stderr: False to suppress stderr from the command.

  Yields:
    Parsed JSON.

  Raises:
    CalledProcessError: cmd returned with a non-zero exit code.
    TimeoutError: cmd has timed out.
  """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None if show_stderr else subprocess.PIPE)
    with _TimeoutThread(p.kill, event_timeout_sec, error_format='No subprocess output for {timeout_sec} seconds') as timer:
        for obj in json_stream.ReadJsonStream(p.stdout):
            timer.Reset()
            yield obj
        p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)