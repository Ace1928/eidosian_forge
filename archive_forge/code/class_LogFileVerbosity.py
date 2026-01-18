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
class LogFileVerbosity(object):
    """A log file verbosity context manager.

  Attributes:
    _context_verbosity: int, The log file verbosity during the context.
    _original_verbosity: int, The original log file verbosity before the
      context was entered.

  Returns:
    The original verbosity is returned in the "as" clause.
  """

    def __init__(self, verbosity):
        self._context_verbosity = verbosity

    def __enter__(self):
        self._original_verbosity = SetLogFileVerbosity(self._context_verbosity)
        return self._original_verbosity

    def __exit__(self, exc_type, exc_value, traceback):
        SetLogFileVerbosity(self._original_verbosity)
        return False