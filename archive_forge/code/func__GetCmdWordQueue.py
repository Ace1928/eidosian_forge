from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def _GetCmdWordQueue(cmd_line):
    """Converts the given cmd_line to a queue of command line words.

  Args:
    cmd_line: str, full command line before parsing.

  Returns:
    [str], Queue of command line words.
  """
    cmd_words = shlex.split(cmd_line)[1:]
    if cmd_line[-1] == _SPACE:
        cmd_words.append(_EMPTY_STRING)
    cmd_words.reverse()
    return cmd_words