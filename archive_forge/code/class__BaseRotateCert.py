from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.api_lib.sql.ssl import server_ca_certs
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
class _BaseRotateCert(object):
    """Base class for sql server_ca_certs rotate."""

    @staticmethod
    def Args(parser):
        """Declare flag and positional arguments for the command parser."""
        base.ASYNC_FLAG.AddToParser(parser)
        flags.AddInstance(parser)
        parser.display_info.AddFormat(flags.SERVER_CA_CERTS_FORMAT)

    def Run(self, args):
        """Rotate in the upcoming server CA cert for a Cloud SQL instance.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      The Server CA Cert that was rotated in, if the operation was successful.
    """
        client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
        sql_client = client.sql_client
        sql_messages = client.sql_messages
        validate.ValidateInstanceName(args.instance)
        instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
        next_server_ca = server_ca_certs.GetNextServerCa(sql_client, sql_messages, instance_ref)
        if not next_server_ca:
            raise exceptions.ResourceNotFoundError('No upcoming Server CA Certificate exists.')
        result_operation = sql_client.instances.RotateServerCa(sql_messages.SqlInstancesRotateServerCaRequest(project=instance_ref.project, instance=instance_ref.instance))
        operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
        operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Rotating to upcoming Server CA Certificate')
        return next_server_ca