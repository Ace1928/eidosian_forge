from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def Subprocess(args, env=None, **extra_popen_kwargs):
    """Run subprocess.Popen with optional timeout and custom env.

  Returns a running subprocess. Depending on the available version of the
  subprocess library, this will return either a subprocess.Popen or a
  SubprocessTimeoutWrapper (which forwards calls to a subprocess.Popen).
  Callers should catch TIMEOUT_EXPIRED_ERR instead of
  subprocess.TimeoutExpired to be compatible with both classes.

  Args:
    args: [str], The arguments to execute.  The first argument is the command.
    env: {str: str}, An optional environment for the child process.
    **extra_popen_kwargs: Any additional kwargs will be passed through directly
      to subprocess.Popen

  Returns:
    subprocess.Popen or SubprocessTimeoutWrapper, The running subprocess.

  Raises:
    PermissionError: if user does not have execute permission for cloud sdk bin
    files.
    InvalidCommandError: if the command entered cannot be found.
  """
    try:
        if args and isinstance(args, list):
            args = [encoding.Encode(a) for a in args]
        p = subprocess.Popen(args, env=GetToolEnv(env=env), **extra_popen_kwargs)
    except OSError as err:
        if err.errno == errno.EACCES:
            raise PermissionError(err.strerror)
        elif err.errno == errno.ENOENT:
            raise InvalidCommandError(args[0])
        raise
    process_holder = _ProcessHolder()
    process_holder.process = p
    if process_holder.signum is not None:
        if p.poll() is None:
            p.terminate()
    try:
        return SubprocessTimeoutWrapper(p)
    except NameError:
        return p