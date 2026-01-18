from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _FmtString(fmt):
    """Gets the correct format string to use based on the Python version.

  Args:
    fmt: text string, The format string to convert.

  Returns:
    A byte string on Python 2 or the original string on Python 3.
  """
    if six.PY2:
        return fmt.encode('utf-8')
    return fmt