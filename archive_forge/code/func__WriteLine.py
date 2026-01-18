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
def _WriteLine(self, line):
    """Writes an encoded line to the coshell."""
    self._shell.communicate(self._Encode(line + '\n'))