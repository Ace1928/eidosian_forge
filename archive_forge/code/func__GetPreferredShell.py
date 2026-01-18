from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import shutil
import sys
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _GetPreferredShell(path, default=_DEFAULT_SHELL):
    """Returns the preferred shell name based on the base file name in path.

  Args:
    path: str, The file path to check.
    default: str, The default value to return if a preferred name cannot be
      determined.

  Returns:
    The preferred user shell name or default if none can be determined.
  """
    if not path:
        return default
    name = os.path.basename(path)
    for shell in _SUPPORTED_SHELLS:
        if shell in six.text_type(name):
            return shell
    return default