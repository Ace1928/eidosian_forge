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
def _UpdateOracleConnectionProfile(self, connection_profile, args, update_fields):
    """Updates PostgreSQL connection profile."""
    if args.IsSpecified('host'):
        connection_profile.oracle.host = args.host
        update_fields.append('oracle.host')
    if args.IsSpecified('port'):
        connection_profile.oracle.port = args.port
        update_fields.append('oracle.port')
    if args.IsSpecified('username'):
        connection_profile.oracle.username = args.username
        update_fields.append('oracle.username')
    if args.IsSpecified('password'):
        connection_profile.oracle.password = args.password
        update_fields.append('oracle.password')
    if args.IsSpecified('database-service'):
        connection_profile.oracle.databaseService = args.databaseService
        update_fields.append('oracle.databaseService')
    self._UpdateOracleSslConfig(connection_profile, args, update_fields)