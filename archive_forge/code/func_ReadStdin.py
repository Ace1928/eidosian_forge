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
def ReadStdin(binary=False):
    """Reads data from stdin, correctly accounting for encoding.

  Anything that needs to read sys.stdin must go through this method.

  Args:
    binary: bool, True to read raw bytes, False to read text.

  Returns:
    A text string if binary is False, otherwise a byte string.
  """
    if binary:
        return files.ReadStdinBytes()
    else:
        data = sys.stdin.read()
        if six.PY2:
            data = console_attr.Decode(data)
        return data