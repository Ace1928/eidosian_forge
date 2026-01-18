from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
def AddPostgresqlProfileGroup(parser, required=True):
    """Adds necessary postgresql profile flags to the given parser."""
    postgresql_profile = parser.add_group()
    postgresql_profile.add_argument('--postgresql-hostname', help='IP or hostname of the PostgreSQL source database.', required=required)
    postgresql_profile.add_argument('--postgresql-port', help='Network port of the PostgreSQL source database.', required=required, type=int)
    postgresql_profile.add_argument('--postgresql-username', help='Username Datastream will use to connect to the database.', required=required)
    postgresql_profile.add_argument('--postgresql-database', help='Database service for the PostgreSQL connection.', required=required)
    password_group = postgresql_profile.add_group(required=required, mutex=True)
    password_group.add_argument('--postgresql-password', help='          Password for the user that Datastream will be using to\n          connect to the database.\n          This field is not returned on request, and the value is encrypted\n          when stored in Datastream.')
    password_group.add_argument('--postgresql-prompt-for-password', action='store_true', help='Prompt for the password used to connect to the database.')