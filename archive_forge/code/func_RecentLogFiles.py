from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import getpass
import io
import locale
import os
import platform as system_platform
import re
import ssl
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import http_proxy_setup
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import requests
import six
import urllib3
def RecentLogFiles(logs_dir, num=1):
    """Finds the most recent (not current) gcloud log files.

  Args:
    logs_dir: str, The path to the logs directory being used.
    num: the number of log files to find

  Returns:
    A list of full paths to the latest num log files, excluding the current
    log file. If there are fewer than num log files, include all of
    them. They will be in chronological order.
  """
    date_dirs = FilesSortedByName(logs_dir)
    if not date_dirs:
        return []
    found_files = []
    for date_dir in reversed(date_dirs):
        log_files = reversed(FilesSortedByName(date_dir) or [])
        found_files.extend(log_files)
        if len(found_files) >= num + 1:
            return found_files[1:num + 1]
    return found_files[1:]