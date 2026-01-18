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
def Positional(d):
    """Returns a positional object suitable for the calliope.markdown module."""
    positional = type(POSITIONAL_TYPE_NAME, (object,), d)
    positional.help = positional.description
    positional.is_group = False
    positional.is_hidden = False
    positional.is_positional = True
    positional.is_required = positional.nargs != '*'
    positional.dest = positional.value.lower().replace('-', '_')
    positional.metavar = positional.value
    positional.option_strings = []
    try:
        positional.nargs = int(positional.nargs)
    except ValueError:
        pass
    return positional