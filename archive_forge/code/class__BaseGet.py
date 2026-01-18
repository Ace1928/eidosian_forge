from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import cert
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
class _BaseGet(object):
    """Base class for sql ssl client_certs list."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use it to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        parser.add_argument('common_name', help='User supplied name. Constrained to ```[a-zA-Z.-_ ]+```.')
        flags.AddInstance(parser)

    def Run(self, args):
        """Retrieve information about a client cert for a Cloud SQL instance.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      A dict object representing the sslCerts resource if the api request was
      successful.
    """
        client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
        sql_client = client.sql_client
        sql_messages = client.sql_messages
        validate.ValidateInstanceName(args.instance)
        instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
        return cert.GetCertFromName(sql_client, sql_messages, instance_ref, args.common_name)