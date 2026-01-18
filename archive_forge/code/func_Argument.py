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
def Argument(d):
    """Returns an argument object suitable for the calliope.markdown module."""
    if d.get(cli_tree.LOOKUP_IS_POSITIONAL, False):
        return Positional(d)
    if not d.get(cli_tree.LOOKUP_IS_GROUP, False):
        return Flag(d)
    group = type(GROUP_TYPE_NAME, (object,), d)
    group.arguments = [Argument(a) for a in d.get(cli_tree.LOOKUP_ARGUMENTS, [])]
    group.category = None
    group.help = group.description
    group.is_global = False
    group.is_hidden = False
    group.sort_args = True
    group.disable_default_heading = False
    return group