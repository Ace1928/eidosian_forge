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
def _GetSqlServerDatabaseBackups(self, sqlserver_databases, sqlserver_encrypted_databases):
    """Returns the sqlserver database backups list.

    Args:
      sqlserver_databases: The list of databases to be migrated.
      sqlserver_encrypted_databases: JSON/YAML file for encryption settings for
        encrypted databases.

    Raises:
      Error: Empty list item in JSON/YAML file.
      Error: Encrypted Database name not found in database list.
      Error: Invalid JSON/YAML file.
    """
    database_backups = []
    encrypted_databases_list = []
    if sqlserver_encrypted_databases:
        for database in sqlserver_encrypted_databases:
            if database is None:
                raise Error('Empty list item in JSON/YAML file.')
            if database['database'] not in sqlserver_databases:
                raise Error('Encrypted Database name {dbName} not found in database list.'.format(dbName=database['database']))
            try:
                database_backup = encoding.PyValueToMessage(self.messages.SqlServerDatabaseBackup, database)
            except Exception as e:
                raise Error(e)
            encrypted_databases_list.append(database['database'])
            database_backups.append(database_backup)
    for database in sqlserver_databases:
        if database in encrypted_databases_list:
            continue
        database_backups.append(self.messages.SqlServerDatabaseBackup(database=database))
    return database_backups