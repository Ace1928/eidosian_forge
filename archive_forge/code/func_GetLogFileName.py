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
def GetLogFileName(suffix):
    """Returns a new log file name based on the currently active log file.

  Args:
    suffix: str, A suffix to add to the current log file name.

  Returns:
    str, The name of a log file, or None if file logging is not on.
  """
    log_file = _log_manager.current_log_file
    if not log_file:
        return None
    log_filename = os.path.basename(log_file)
    log_file_root_name = log_filename[:-len(LOG_FILE_EXTENSION)]
    return log_file_root_name + suffix