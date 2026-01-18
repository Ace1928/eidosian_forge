from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _GetSqlserverHomogeneousMigrationJobConfig(self, args):
    """Returns the sqlserver homogeneous migration job config.

    Args:
      args: argparse.Namespace, the arguments that this command was invoked
        with.
    """
    sqlserver_homogeneous_migration_job_config_obj = self.messages.SqlServerHomogeneousMigrationJobConfig(backupFilePattern=args.sqlserver_backup_file_pattern)
    if args.IsKnownAndSpecified('sqlserver_databases'):
        sqlserver_homogeneous_migration_job_config_obj.databaseBackups = self._GetSqlServerDatabaseBackups(args.sqlserver_databases, args.sqlserver_encrypted_databases)
    return sqlserver_homogeneous_migration_job_config_obj