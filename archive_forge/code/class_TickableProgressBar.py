from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import enum
import getpass
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_pager
from googlecloudsdk.core.console import prompt_completer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
class TickableProgressBar(object):
    """A thread safe progress bar with a discrete number of tasks."""

    def __init__(self, total, *args, **kwargs):
        self.completed = 0
        self.total = total
        self._progress_bar = ProgressBar(*args, **kwargs)
        self._lock = threading.Lock()

    def __enter__(self):
        self._progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._progress_bar.__exit__(exc_type, exc_value, traceback)

    def Tick(self):
        with self._lock:
            self.completed += 1
            self._progress_bar.SetProgress(self.completed / self.total)