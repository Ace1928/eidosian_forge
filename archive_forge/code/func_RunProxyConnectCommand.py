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
def RunProxyConnectCommand(args, supports_database=False):
    """Connects to a Cloud SQL instance through the Cloud SQL Proxy.

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
    CloudSqlProxyError: Cloud SQL Proxy could not be found.
    SqlClientNotFoundError: A local SQL client could not be found.
    ConnectionError: An error occurred while trying to connect to the instance.
  """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    instance_ref = instances_command_util.GetInstanceRef(args, client)
    instance_info = sql_client.instances.Get(sql_messages.SqlInstancesGetRequest(project=instance_ref.project, instance=instance_ref.instance))
    if not instances_api_util.IsInstanceV2(sql_messages, instance_info):
        return RunConnectCommand(args, supports_database)
    exe = files.FindExecutableOnPath('cloud_sql_proxy')
    if not exe:
        raise exceptions.CloudSqlProxyError("Cloud SQL Proxy (v1) couldn't be found in PATH. Either install the component with `gcloud components install cloud_sql_proxy` or see https://github.com/GoogleCloudPlatform/cloud-sql-proxy/releases to install the v1 Cloud SQL Proxy. The v2 Cloud SQL Proxy is currently not supported by the connect command. You need to install the v1 Cloud SQL Proxy binary to use the connect command.")
    db_type = instance_info.databaseVersion.name.split('_')[0]
    exe_name = constants.DB_EXE.get(db_type, 'mysql')
    exe = files.FindExecutableOnPath(exe_name)
    if not exe:
        raise exceptions.SqlClientNotFoundError('{0} client not found.  Please install a {1} client and make sure it is in PATH to be able to connect to the database instance.'.format(exe_name.title(), exe_name))
    port = six.text_type(args.port)
    proxy_process = instances_api_util.StartCloudSqlProxy(instance_info, port)
    atexit.register(proxy_process.kill)
    sql_user = constants.DEFAULT_SQL_USER[exe_name]
    if args.user:
        sql_user = args.user
    flags = constants.EXE_FLAGS[exe_name]
    sql_args = [exe_name]
    if exe_name == 'mssql-cli':
        hostname = 'tcp:127.0.0.1,{0}'.format(port)
        sql_args.extend([flags['hostname'], hostname])
    else:
        sql_args.extend([flags['hostname'], '127.0.0.1', flags['port'], port])
    sql_args.extend([flags['user'], sql_user])
    if 'password' in flags:
        sql_args.append(flags['password'])
    if supports_database:
        sql_args.extend(instances_command_util.GetDatabaseArgs(args, flags))
    instances_command_util.ConnectToInstance(sql_args, sql_user)
    proxy_process.kill()