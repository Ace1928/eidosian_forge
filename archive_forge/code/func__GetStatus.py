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
def _GetStatus(self):
    """Gets the status of the last command sent to the coshell."""
    status_string = self._ReadLine()
    if status_string.endswith(self.SHELL_STATUS_EXIT):
        c = self.SHELL_STATUS_EXIT
        status_string = status_string[:-1]
    else:
        c = ''
    if not status_string.isdigit() or c == self.SHELL_STATUS_EXIT:
        self._Exited()
    return int(status_string)