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
def _IsTaskKillError(stderr):
    """Returns whether the stderr output of taskkill indicates it failed.

  Args:
    stderr: the string error output of the taskkill command

  Returns:
    True iff the stderr is considered to represent an actual error.
  """
    non_error_reasons = ('Access is denied.', 'The operation attempted is not supported.', 'There is no running instance of the task.', 'There is no running instance of the task to terminate.')
    non_error_patterns = (re.compile('The process "\\d+" not found\\.'),)
    for reason in non_error_reasons:
        if reason in stderr:
            return False
    for pattern in non_error_patterns:
        if pattern.search(stderr):
            return False
    return True