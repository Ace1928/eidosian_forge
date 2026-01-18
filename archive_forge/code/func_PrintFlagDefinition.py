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
def PrintFlagDefinition(self, flag, disable_header=False):
    """Prints a flags definition list item."""
    if isinstance(flag, dict):
        flag = Flag(flag)
    super(CliTreeMarkdownGenerator, self).PrintFlagDefinition(flag, disable_header=disable_header)