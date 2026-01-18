from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddDatabaseVersionFlag(parser):
    """Adds a --database-version flag to the given parser."""
    help_text = 'Database engine major version.'
    choices = ['POSTGRES_14', 'POSTGRES_15']
    parser.add_argument('--database-version', help=help_text, choices=choices)