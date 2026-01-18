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
def IsValidSubPath(self, command_path):
    """Returns True if the given command path after the top is valid."""
    return self._GetCommandFromPath([cli_tree.DEFAULT_CLI_NAME] + command_path) is not None