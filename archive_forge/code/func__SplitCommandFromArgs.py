from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _SplitCommandFromArgs(self, cmd):
    """Splits cmd into command and args lists.

    The command list part is a valid command and the args list part is the
    trailing args.

    Args:
      cmd: [str], A command + args list.

    Returns:
      (command, args): The command and args lists.
    """
    if len(cmd) <= 1:
        return (cmd, [])
    skip = 1
    i = skip
    while i <= len(cmd):
        i += 1
        if not self.IsValidSubPath(cmd[skip:i]):
            i -= 1
            break
    return (cmd[:i], cmd[i:])