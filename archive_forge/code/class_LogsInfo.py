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
class LogsInfo(object):
    """Holds information about where logs are located."""
    NUM_RECENT_LOG_FILES = 5

    def __init__(self, anonymizer=None):
        anonymizer = anonymizer or NoopAnonymizer()
        paths = config.Paths()
        logs_dir = paths.logs_dir
        self.last_log = anonymizer.ProcessPath(LastLogFile(logs_dir))
        self.last_logs = [anonymizer.ProcessPath(f) for f in RecentLogFiles(logs_dir, self.NUM_RECENT_LOG_FILES)]
        self.logs_dir = anonymizer.ProcessPath(logs_dir)

    def __str__(self):
        return textwrap.dedent('        Logs Directory: [{logs_dir}]\n        Last Log File: [{log_file}]\n        '.format(logs_dir=self.logs_dir, log_file=self.last_log))

    def LastLogContents(self):
        last_log = LastLogFile(config.Paths().logs_dir)
        if not self.last_log:
            return ''
        return file_utils.ReadFileContents(last_log)

    def GetRecentRuns(self):
        """Return the most recent runs, as reported by info_holder.LogsInfo.

    Returns:
      A list of LogData
    """
        last_logs = RecentLogFiles(config.Paths().logs_dir, self.NUM_RECENT_LOG_FILES)
        return [LogData.FromFile(log_file) for log_file in last_logs]