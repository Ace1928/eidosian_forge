from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import subprocess
import sys
from googlecloudsdk.core.util import files
def ExecuteCommand(cmd, input_str=None, file=None):
    """Executes shell commands in subprocess.

  Executes the supplied command with the supplied standard input string, streams
  the output to stdout, and returns the process's return code.

  Args:
    cmd: (List[str]) Strings to send in as the command.
    input_str: (str) if supplied, it will be passed as stdin to the supplied
      command. if None, stdin will get closed immediately.
    file: optional file-like object (stream), the output from the executed
      process's stdout will get sent to this stream. Defaults to sys.stdout.

  Returns:
    return code of the process
  """
    if file is None:
        file = sys.stdout
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=False, bufsize=1) as p:
        if input_str:
            p.stdin.write(input_str.encode('utf-8'))
        p.stdin.close()
        out = io.TextIOWrapper(p.stdout, newline='')
        for line in out:
            file.write(line)
            file.flush()
        else:
            file.flush()
    return p.returncode