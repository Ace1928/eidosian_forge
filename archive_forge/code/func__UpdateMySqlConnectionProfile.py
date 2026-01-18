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
def _UpdateMySqlConnectionProfile(self, connection_profile, args, update_fields):
    """Updates MySQL connection profile."""
    if args.IsSpecified('host'):
        connection_profile.mysql.host = args.host
        update_fields.append('mysql.host')
    if args.IsSpecified('port'):
        connection_profile.mysql.port = args.port
        update_fields.append('mysql.port')
    if args.IsSpecified('username'):
        connection_profile.mysql.username = args.username
        update_fields.append('mysql.username')
    if args.IsSpecified('password'):
        connection_profile.mysql.password = args.password
        update_fields.append('mysql.password')
    if args.IsSpecified(self._InstanceArgName()):
        connection_profile.mysql.cloudSqlId = args.GetValue(self._InstanceArgName())
        update_fields.append('mysql.instance')
    self._UpdateMySqlSslConfig(connection_profile, args, update_fields)