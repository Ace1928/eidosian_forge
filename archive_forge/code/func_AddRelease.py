from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddRelease(parser, help_text, hidden=False):
    """Adds release flag."""
    parser.add_argument('--release', hidden=hidden, help=help_text)