from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
def _GetSuffix(self):
    if self.spinner_override_message:
        num_dots = self._ticks % 4
        return self.spinner_override_message + '.' * num_dots
    else:
        return self._symbols.spin_marks[self._ticks % len(self._symbols.spin_marks)]