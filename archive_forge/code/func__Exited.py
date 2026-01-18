from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def _Exited(self):
    """Raises the coshell exit exception."""
    try:
        self._WriteLine(':')
    except (IOError, OSError, ValueError):
        pass
    status = self._ShellStatus(self._shell.returncode)
    raise CoshellExitError('The coshell exited [status={}].'.format(status), status=status)