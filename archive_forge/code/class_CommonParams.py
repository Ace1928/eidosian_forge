from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
class CommonParams(object):
    """Parameters common to all metrics reporters."""

    def __init__(self):
        hostname = socket.gethostname()
        install_type = 'Google' if hostname.endswith('.google.com') else 'External'
        current_platform = platforms.Platform.Current()
        self.client_id = config.GetCID()
        self.current_platform = current_platform
        self.user_agent = GetUserAgent(current_platform)
        self.release_channel = config.INSTALLATION_CONFIG.release_channel
        self.install_type = install_type
        self.metrics_environment = properties.GetMetricsEnvironment()
        self.is_interactive = console_io.IsInteractive(error=True, heuristic=True)
        self.python_version = platform.python_version()
        self.metrics_environment_version = properties.VALUES.metrics.environment_version.Get()
        self.is_run_from_shell_script = console_io.IsRunFromShellScript()
        self.term_identifier = console_attr.GetConsoleAttr().GetTermIdentifier()