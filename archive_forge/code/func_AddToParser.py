from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def AddToParser(self, parser):
    """Adds all attributes of the concept to argparse.

    Creates a group to hold all the attributes and adds an argument for each
    attribute. If the presentation spec is required, then the anchor attribute
    argument will be required.

    Args:
      parser: the parser for the Calliope command.
    """
    args = self.GetAttributeArgs()
    if not args:
        return
    parser = self.group or parser
    hidden = any((x.IsHidden() for x in args))
    resource_group = parser.add_group(help=self.GetGroupHelp(), required=self.args_required, hidden=hidden)
    for arg in args:
        arg.AddToParser(resource_group)