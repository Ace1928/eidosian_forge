from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddDumpTypeFlag(parser):
    """Adds a --dump-type flag to the given parser."""
    help_text = 'The type of the data dump. Currently applicable for MySQL to MySQL migrations only.'
    choices = ['LOGICAL', 'PHYSICAL']
    parser.add_argument('--dump-type', help=help_text, choices=choices)