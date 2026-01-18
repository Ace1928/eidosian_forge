from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _DetectGems(bundler_available):
    """Returns a list of gems requested by this application.

  Args:
    bundler_available: (bool) Whether bundler is available in the environment.

  Returns:
    ([str, ...]) A list of gem names.
  """
    gems = []
    if bundler_available:
        for line in _RunSubprocess('bundle list').splitlines():
            match = re.match('\\s*\\*\\s+(\\S+)\\s+\\(', line)
            if match:
                gems.append(match.group(1))
    return gems