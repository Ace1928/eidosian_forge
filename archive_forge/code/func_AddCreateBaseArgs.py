from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCreateBaseArgs(parser):
    """Adds common arguments for creating a network."""
    parser.add_argument('--description', help='An optional, textual description for the network.')
    parser.add_argument('--range', help=RANGE_HELP_TEXT)