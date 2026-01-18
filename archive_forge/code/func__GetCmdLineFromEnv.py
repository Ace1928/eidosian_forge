from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def _GetCmdLineFromEnv():
    """Gets the command line from the environment.

  Returns:
    str, Command line.
  """
    cmd_line = encoding.GetEncodedValue(os.environ, LINE_ENV_VAR)
    completion_point = int(encoding.GetEncodedValue(os.environ, POINT_ENV_VAR))
    cmd_line = cmd_line[:completion_point]
    return cmd_line