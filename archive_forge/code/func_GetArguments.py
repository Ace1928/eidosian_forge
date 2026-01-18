from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
import six
def GetArguments(self):
    """Returns the command arguments."""
    command = self._GetCommandFromPath(self._command_path)
    try:
        return [Argument(a) for a in command[cli_tree.LOOKUP_CONSTRAINTS][cli_tree.LOOKUP_ARGUMENTS]]
    except (KeyError, TypeError):
        return []