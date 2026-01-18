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
def ReadFromFileOrStdin(path, binary):
    """Returns the contents of the specified file or stdin if path is '-'.

  Args:
    path: str, The path of the file to read.
    binary: bool, True to open the file in binary mode.

  Raises:
    Error: If the file cannot be read or is larger than max_bytes.

  Returns:
    The contents of the file.
  """
    if path == '-':
        return ReadStdin(binary=binary)
    if binary:
        return files.ReadBinaryFileContents(path)
    return files.ReadFileContents(path)