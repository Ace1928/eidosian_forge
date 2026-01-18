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
def _StreamSubprocessOutput(proc, raw=False, stdout_handler=log.Print, stderr_handler=log.status.Print, capture=False):
    """Log stdout and stderr output from running sub-process."""
    stdout = []
    stderr = []
    with ReplaceEnv(PYTHONUNBUFFERED='1'):
        while True:
            out_line = proc.stdout.readline() or b''
            err_line = proc.stderr.readline() or b''
            if not (err_line or out_line) and proc.poll() is not None:
                break
            if out_line:
                if capture:
                    stdout.append(out_line)
                out_str = out_line.decode('utf-8')
                out_str = out_str.rstrip('\r\n') if not raw else out_str
                stdout_handler(out_str)
            if err_line:
                if capture:
                    stderr.append(err_line)
                err_str = err_line.decode('utf-8')
                err_str = err_str.rstrip('\r\n') if not raw else err_str
                stderr_handler(err_str)
    return (proc.returncode, stdout, stderr)