from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _UpdatePostgresqlProfile(self, connection_profile, args, update_fields):
    """Updates Postgresql connection profile."""
    if args.IsSpecified('postgresql_hostname'):
        connection_profile.postgresqlProfile.hostname = args.postgresql_hostname
        update_fields.append('postgresqlProfile.hostname')
    if args.IsSpecified('postgresql_port'):
        connection_profile.postgresqlProfile.port = args.postgresql_port
        update_fields.append('postgresqlProfile.port')
    if args.IsSpecified('postgresql_username'):
        connection_profile.postgresqlProfile.username = args.postgresql_username
        update_fields.append('postgresqlProfile.username')
    if args.IsSpecified('postgresql_password'):
        connection_profile.postgresqlProfile.password = args.postgresql_password
        update_fields.append('postgresqlProfile.password')
    if args.IsSpecified('postgresql_database'):
        connection_profile.postgresqlProfile.database = args.postgresql_database
        update_fields.append('postgresqlProfile.database')