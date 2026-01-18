from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import import_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def RunSqlImportCommand(args, client):
    """Imports data from a SQL dump file into Cloud SQL instance.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    client: SqlClient instance, with sql_client and sql_messages props, for use
      in generating messages and making API calls.

  Returns:
    A dict representing the import operation resource, if '--async' is used,
    or else None.
  """
    sql_import_context = import_util.SqlImportContext(client.sql_messages, args.uri, args.database, args.user, parallel=args.parallel, threads=args.threads)
    return RunImportCommand(args, client, sql_import_context)