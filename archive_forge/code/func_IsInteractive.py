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
def IsInteractive(output=False, error=False, heuristic=False):
    """Determines if the current terminal session is interactive.

  sys.stdin must be a terminal input stream.

  Args:
    output: If True then sys.stdout must also be a terminal output stream.
    error: If True then sys.stderr must also be a terminal output stream.
    heuristic: If True then we also do some additional heuristics to check if
               we are in an interactive context. Checking home path for example.

  Returns:
    True if the current terminal session is interactive.
  """
    try:
        if not sys.stdin.isatty():
            return False
        if output and (not sys.stdout.isatty()):
            return False
        if error and (not sys.stderr.isatty()):
            return False
    except AttributeError:
        return False
    if heuristic:
        home = encoding.GetEncodedValue(os.environ, 'HOME')
        homepath = encoding.GetEncodedValue(os.environ, 'HOMEPATH')
        if not homepath and (not home or home == '/'):
            return False
    return True