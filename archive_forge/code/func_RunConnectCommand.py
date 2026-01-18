from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances as instances_api_util
from googlecloudsdk.api_lib.sql import network
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.sql import flags as sql_flags
from googlecloudsdk.command_lib.sql import instances as instances_command_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
import six
import six.moves.http_client
def RunConnectCommand(args, supports_database=False):
    """Connects to a Cloud SQL instance directly.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    supports_database: Whether or not the `--database` flag needs to be
      accounted for.

  Returns:
    If no exception is raised this method does not return. A new process is
    started and the original one is killed.
  Raises:
    HttpException: An http error response was received while executing api
        request.
    UpdateError: An error occurred while updating an instance.
    SqlClientNotFoundError: A local SQL client could not be found.
    ConnectionError: An error occurred while trying to connect to the instance.
  """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    instance_ref = instances_command_util.GetInstanceRef(args, client)
    acl_name = _AllowlistClientIP(instance_ref, sql_client, sql_messages, client.resource_parser)
    retryer = retry.Retryer(max_retrials=2, exponential_sleep_multiplier=2)
    try:
        instance_info, client_ip = retryer.RetryOnResult(_GetClientIP, [instance_ref, sql_client, acl_name], should_retry_if=lambda x, s: x[1] is None, sleep_ms=500)
    except retry.RetryException:
        raise exceptions.UpdateError('Could not allowlist client IP. Server did not reply with the allowlisted IP.')
    db_type = instance_info.databaseVersion.name.split('_')[0]
    exe_name = constants.DB_EXE.get(db_type, 'mysql')
    exe = files.FindExecutableOnPath(exe_name)
    if not exe:
        raise exceptions.SqlClientNotFoundError('{0} client not found.  Please install a {1} client and make sure it is in PATH to be able to connect to the database instance.'.format(exe_name.title(), exe_name))
    ip_type = network.GetIpVersion(client_ip)
    if ip_type == network.IP_VERSION_4:
        if instance_info.settings.ipConfiguration.ipv4Enabled:
            ip_address = instance_info.ipAddresses[0].ipAddress
        else:
            message = 'It seems your client does not have ipv6 connectivity and the database instance does not have an ipv4 address. Please request an ipv4 address for this database instance.'
            raise exceptions.ConnectionError(message)
    elif ip_type == network.IP_VERSION_6:
        ip_address = instance_info.ipv6Address
    else:
        raise exceptions.ConnectionError('Could not connect to SQL server.')
    sql_user = constants.DEFAULT_SQL_USER[exe_name]
    if args.user:
        sql_user = args.user
    flags = constants.EXE_FLAGS[exe_name]
    sql_args = [exe_name, flags['hostname'], ip_address]
    sql_args.extend([flags['user'], sql_user])
    if 'password' in flags:
        sql_args.append(flags['password'])
    if supports_database:
        sql_args.extend(instances_command_util.GetDatabaseArgs(args, flags))
    instances_command_util.ConnectToInstance(sql_args, sql_user)