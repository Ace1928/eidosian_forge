from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddPasswordFlagGroup(parser, required=False):
    """Adds --password and --prompt-for-password flags to the given parser."""
    password_group = parser.add_group(required=required, mutex=True)
    password_group.add_argument('--password', help='          Password for the user that Database Migration Service uses to\n          connect to the database. Database Migration Service encrypts\n          the value when storing it, and the field is not returned on request.\n          ')
    password_group.add_argument('--prompt-for-password', action='store_true', help='Prompt for the password used to connect to the database.')