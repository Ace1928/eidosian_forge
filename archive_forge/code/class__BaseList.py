from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.api_lib.sql.ssl import server_ca_certs
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
class _BaseList(object):
    """Base class for sql ssl server_ca_certs list."""

    @staticmethod
    def Args(parser):
        flags.AddInstance(parser)
        parser.display_info.AddFormat(flags.SERVER_CA_CERTS_FORMAT)

    def Run(self, args):
        """List all server CA certs for a Cloud SQL instance.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      A dict object that has the list of sslCerts resources if the api request
      was successful.
    """
        client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
        sql_client = client.sql_client
        sql_messages = client.sql_messages
        validate.ValidateInstanceName(args.instance)
        instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
        result = server_ca_certs.ListServerCas(sql_client, sql_messages, instance_ref)
        return iter(result.certs)