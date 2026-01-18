from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def ReplicaConfiguration(sql_messages, primary_username, primary_password, primary_dump_file_path, primary_ca_certificate_path=None, client_certificate_path=None, client_key_path=None):
    """Generates the config for an external primary replica.

  Args:
    sql_messages: module, The messages module that should be used.
    primary_username: The username for connecting to the external instance.
    primary_password: The password for connecting to the external instance.
    primary_dump_file_path: ObjectReference, a wrapper for the URI of the Cloud
      Storage path containing the dumpfile to seed the replica with.
    primary_ca_certificate_path: The path to the CA certificate PEM file.
    client_certificate_path: The path to the client certificate PEM file.
    client_key_path: The path to the client private key PEM file.

  Returns:
    sql_messages.MySqlReplicaConfiguration object.
  """
    mysql_replica_configuration = sql_messages.MySqlReplicaConfiguration(kind='sql#mysqlReplicaConfiguration', username=primary_username, password=primary_password, dumpFilePath=primary_dump_file_path.ToUrl())
    if primary_ca_certificate_path:
        mysql_replica_configuration.caCertificate = files.ReadFileContents(primary_ca_certificate_path)
    if client_certificate_path:
        mysql_replica_configuration.clientCertificate = files.ReadFileContents(client_certificate_path)
    if client_key_path:
        mysql_replica_configuration.clientKey = files.ReadFileContents(client_key_path)
    return sql_messages.ReplicaConfiguration(kind='sql#demoteMasterMysqlReplicaConfiguration', mysqlReplicaConfiguration=mysql_replica_configuration)