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
def GetLogDir():
    """Gets the path to the currently in use log directory.

  Returns:
    str, The logging directory path.
  """
    log_file = _log_manager.current_log_file
    if not log_file:
        return None
    return os.path.dirname(log_file)