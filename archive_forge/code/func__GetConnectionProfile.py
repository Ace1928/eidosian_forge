from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetConnectionProfile(self, cp_type, args, connection_profile_id):
    """Returns a connection profile according to type."""
    connection_profile_type = self.messages.ConnectionProfile
    labels = labels_util.ParseCreateArgs(args, connection_profile_type.LabelsValue)
    params = {}
    if cp_type == 'MYSQL':
        mysql_connection_profile = self._GetMySqlConnectionProfile(args)
        params['mysql'] = mysql_connection_profile
        params['provider'] = self._GetProvider(connection_profile_type, args.provider)
    elif cp_type == 'CLOUDSQL':
        cloudsql_connection_profile = self._GetCloudSqlConnectionProfile(args)
        params['cloudsql'] = cloudsql_connection_profile
        params['provider'] = self._GetProvider(connection_profile_type, args.provider)
    elif cp_type == 'POSTGRESQL':
        postgresql_connection_profile = self._GetPostgreSqlConnectionProfile(args)
        params['postgresql'] = postgresql_connection_profile
    elif cp_type == 'ALLOYDB':
        alloydb_connection_profile = self._GetAlloyDBConnectionProfile(args, connection_profile_id)
        params['alloydb'] = alloydb_connection_profile
    elif cp_type == 'ORACLE':
        oracle_connection_profile = self._GetOracleConnectionProfile(args)
        params['oracle'] = oracle_connection_profile
    elif cp_type == 'SQLSERVER':
        sqlserver_connection_profile = self._GetSqlServerConnectionProfile(args)
        params['sqlserver'] = sqlserver_connection_profile
        params['provider'] = self._GetProvider(connection_profile_type, args.provider)
    return connection_profile_type(labels=labels, state=connection_profile_type.StateValueValuesEnum.CREATING, displayName=args.display_name, **params)