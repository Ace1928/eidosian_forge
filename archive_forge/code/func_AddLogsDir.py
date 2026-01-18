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
def AddLogsDir(self, logs_dir):
    """Adds a new logging directory and configures file logging.

    Args:
      logs_dir: str, Path to a directory to store log files under.  This method
        has no effect if this is None, or if this directory has already been
        registered.
    """
    if not logs_dir or logs_dir in self._logs_dirs:
        return
    self._logs_dirs.append(logs_dir)
    self._CleanUpLogs(logs_dir)
    if properties.VALUES.core.disable_file_logging.GetBool():
        return
    try:
        log_file = self._SetupLogsDir(logs_dir)
        file_handler = logging.FileHandler(log_file, encoding=LOG_FILE_ENCODING)
    except (OSError, IOError, files.Error) as exp:
        warning('Could not setup log file in {0}, ({1}: {2}.\nThe configuration directory may not be writable. To learn more, see https://cloud.google.com/sdk/docs/configurations#creating_a_configuration'.format(logs_dir, type(exp).__name__, exp))
        return
    self.current_log_file = log_file
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(self._file_formatter)
    self._root_logger.addHandler(file_handler)
    self.file_only_logger.addHandler(file_handler)