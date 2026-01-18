from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSourceFlag():
    """Adds source flag."""
    return base.Argument('--source', help=_SOURCE_HELP_TEXT, default='.')