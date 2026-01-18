from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddUsernameFlag(parser, required=False):
    """Adds a --username flag to the given parser."""
    help_text = '    Username that Database Migration Service uses to connect to the\n    database. Database Migration Service encrypts the value when storing it.\n    '
    parser.add_argument('--username', help=help_text, required=required)