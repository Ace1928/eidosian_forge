from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
import random
import re
import socket
import subprocess
import tempfile
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
@contextlib.contextmanager
def Exec(args, log_file=None):
    """Starts subprocess with given args and ensures its termination upon exit.

  This starts a subprocess with the given args. The stdout and stderr of the
  subprocess are piped. Note that this is a context manager, to ensure that
  processes (and references to them) are not leaked.

  Args:
    args: [str], The arguments to execute. The first argument is the command.
    log_file: optional file argument to reroute process's output. If given,
      will be closed when the file is terminated.

  Yields:
    process, The process handle of the subprocess that has been started.
  """
    reroute_stdout = log_file or subprocess.PIPE
    if not platforms.OperatingSystem.IsWindows():
        if os.getsid(0) != os.getpid():
            os.setpgid(0, 0)
    process = subprocess.Popen(args, stdout=reroute_stdout, stderr=subprocess.STDOUT)
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()