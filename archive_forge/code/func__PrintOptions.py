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
def _PrintOptions(options, write, limit=None):
    """Prints the options provided to stderr.

  Args:
    options:  [object], A list of objects to print as choices.  Their str()
      method will be used to display them.
    write: f(x)->None, A function to call to write the data.
    limit: int, If set, will only print the first number of options equal
      to the given limit.
  """
    limited_options = options if limit is None else options[:limit]
    for i, option in enumerate(limited_options):
        write(' [{index}] {option}\n'.format(index=i + 1, option=six.text_type(option)))