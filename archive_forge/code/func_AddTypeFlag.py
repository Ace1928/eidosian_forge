from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddTypeFlag(parser, required=False):
    """Adds --type flag to the given parser."""
    help_text = 'Type of the migration job.'
    choices = ['ONE_TIME', 'CONTINUOUS']
    parser.add_argument('--type', help=help_text, choices=choices, required=required)