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
def _KillProcIfRunning(proc):
    """Kill process and close open streams."""
    if proc:
        code = None
        if hasattr(proc, 'returncode'):
            code = proc.returncode
        elif hasattr(proc, 'exitcode'):
            code = proc.exitcode
        if code is None or proc.poll() is None:
            proc.terminate()
        try:
            if proc.stdin and (not proc.stdin.closed):
                proc.stdin.close()
            if proc.stdout and (not proc.stdout.closed):
                proc.stdout.close()
            if proc.stderr and (not proc.stderr.closed):
                proc.stderr.close()
        except OSError:
            pass