from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddSqlServerDatabasesFlag(parser):
    """Adds a --sqlserver-databases flag to the given parser."""
    help_text = '    A list of databases to be migrated to the destination Cloud SQL instance.\n    Provide databases as a comma separated list. This list should contain all\n    encrypted and non-encrypted database names.\n    '
    parser.add_argument('--sqlserver-databases', metavar='databaseName', type=arg_parsers.ArgList(min_length=1), help=help_text, required=True)