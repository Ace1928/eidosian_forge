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
def GetArgDetails(self, arg, depth=None):
    """Returns the help text with auto-generated details for arg.

    The help text was already generated on the cli_tree generation side.

    Args:
      arg: The arg to auto-generate help text for.
      depth: The indentation depth at which the details should be printed.
        Added here only to maintain consistency with superclass during testing.

    Returns:
      The help text with auto-generated details for arg.
    """
    return arg.help