from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class AddDatabase(base.Command):
    """Creates a database for a Cloud SQL instance."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        flags.AddDatabaseName(parser)
        flags.AddCharset(parser)
        flags.AddCollation(parser)
        flags.AddInstance(parser)
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        """Creates a database for a Cloud SQL instance.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      A dict object representing the operations resource describing the create
      operation if the create was successful.
    """
        client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
        sql_client = client.sql_client
        sql_messages = client.sql_messages
        validate.ValidateInstanceName(args.instance)
        instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
        new_database = sql_messages.Database(kind='sql#database', project=instance_ref.project, instance=instance_ref.instance, name=args.database, charset=args.charset, collation=args.collation)
        result_operation = sql_client.databases.Insert(new_database)
        operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
        if args.async_:
            result = sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
        else:
            try:
                operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Creating Cloud SQL database')
            except exceptions.OperationError:
                log.Print('Database creation failed. Check if a database named {0} already exists.'.format(args.database))
                raise
            result = new_database
            result.kind = None
        log.CreatedResource(args.database, kind='database', is_async=args.async_)
        return result