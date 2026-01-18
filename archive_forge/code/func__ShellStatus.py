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
@staticmethod
def _ShellStatus(status):
    """Returns the shell $? status given a python Popen returncode."""
    if status is None:
        status = 0
    elif status < 0:
        status = 256 - status
    return status