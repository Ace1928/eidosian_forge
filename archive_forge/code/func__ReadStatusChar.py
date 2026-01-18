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
def _ReadStatusChar(self):
    """Reads and returns one encoded character from the coshell status fd."""
    return os.read(self._status_fd, 1)