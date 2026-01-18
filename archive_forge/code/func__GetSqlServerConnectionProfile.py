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
def _GetSqlServerConnectionProfile(self, args):
    """Creates an SQL Server connection profile according to the given args.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      SqlServerConnectionProfile, to use when creating the connection profile.
    """
    connection_profile_obj = self.messages.SqlServerConnectionProfile()
    if args.IsKnownAndSpecified('cloudsql_instance'):
        connection_profile_obj.cloudSqlId = args.GetValue(self._InstanceArgName())
    else:
        connection_profile_obj.backups = self._GetSqlServerBackups(args)
    return connection_profile_obj