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
class SubprocessTimeoutWrapper:
    """Forwarding wrapper for subprocess.Popen, adds timeout arg to wait.

    subprocess.Popen.wait doesn't provide a timeout in versions < 3.3. This
    class wraps subprocess.Popen, adds a backported wait that includes the
    timeout arg, and forwards other calls to the underlying subprocess.Popen.

    Callers generally shouldn't use this class directly: Subprocess will
    return either a subprocess.Popen or SubprocessTimeoutWrapper as
    appropriate based on the available version of subprocesses.

    See
    https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait.
    """

    def __init__(self, proc):
        self.proc = proc

    def wait(self, timeout=None):
        """Busy-wait for wrapped process to have a return code.

      Args:
        timeout: int, Seconds to wait before raising TimeoutExpired.

      Returns:
        int, The subprocess return code.

      Raises:
        TimeoutExpired: if subprocess doesn't finish before the given timeout.
      """
        if timeout is None:
            return self.proc.wait()
        now = time.time()
        later = now + timeout
        delay = 0.01
        ret = self.proc.poll()
        while ret is None:
            if time.time() > later:
                raise TimeoutExpired()
            time.sleep(delay)
            ret = self.proc.poll()
        return ret

    def __getattr__(self, name):
        return getattr(self.proc, name)