from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _ClientConnectionConfig(alloydb_messages, ssl_mode=None, require_connectors=None):
    """Generates the client connection config for the instance.

  Args:
    alloydb_messages: module, Message module for the API client.
    ssl_mode: string, SSL mode to use when connecting to the database.
    require_connectors: boolean, whether or not to enforce connections to the
      database to go through a connector (ex: Auth Proxy).

  Returns:
    alloydb_messages.ClientConnectionConfig
  """
    should_generate_config = any([ssl_mode is not None, require_connectors is not None])
    if not should_generate_config:
        return None
    client_connection_config = alloydb_messages.ClientConnectionConfig()
    client_connection_config.requireConnectors = require_connectors
    ssl_config = alloydb_messages.SslConfig()
    ssl_config.sslMode = _ParseSSLMode(alloydb_messages, ssl_mode)
    client_connection_config.sslConfig = ssl_config
    return client_connection_config