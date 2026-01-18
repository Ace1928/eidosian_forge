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
def FormatExample(self, cmd, args, with_args):
    """Creates a link to the command reference from a command example.

    If with_args is False and the provided command includes args,
    returns None.

    Args:
      cmd: [str], a command.
      args: [str], args with the command.
      with_args: bool, whether the example is valid if it has args.

    Returns:
      (str) a representation of the command with a link to the reference, plus
      any args. | None, if the command isn't valid.
    """
    if args and (not with_args):
        return None
    ref = '/'.join(cmd)
    command_link = 'link:' + ref + '[' + ' '.join(cmd) + ']'
    if args:
        command_link += ' ' + ' '.join(args)
    return command_link