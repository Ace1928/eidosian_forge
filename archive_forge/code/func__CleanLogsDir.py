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
def _CleanLogsDir(self, logs_dir):
    """Cleans up old log files form the given logs directory.

    Args:
      logs_dir: str, The path to the logs directory.
    """
    now = datetime.datetime.now()
    now_seconds = time.time()
    try:
        dirnames = os.listdir(logs_dir)
    except (OSError, UnicodeError):
        return
    for dirname in dirnames:
        dir_path = os.path.join(logs_dir, dirname)
        if self._ShouldDeleteDir(now, dir_path):
            for filename in os.listdir(dir_path):
                log_file_path = os.path.join(dir_path, filename)
                if self._ShouldDeleteFile(now_seconds, log_file_path):
                    os.remove(log_file_path)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass