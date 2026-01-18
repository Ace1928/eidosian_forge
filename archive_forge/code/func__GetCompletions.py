from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def _GetCompletions():
    """Returns the static completions, None if there are none."""
    root = LoadCompletionCliTree()
    cmd_line = _GetCmdLineFromEnv()
    return _FindCompletions(root, cmd_line)