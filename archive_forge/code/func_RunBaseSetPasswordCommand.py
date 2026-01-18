from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import users
from googlecloudsdk.core import properties
def RunBaseSetPasswordCommand(args):
    """Changes a user's password in a given instance.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked
      with.

  Returns:
    SQL user resource iterator.
  """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    operation_ref = None
    user = sql_client.users.Get(sql_messages.SqlUsersGetRequest(project=instance_ref.project, instance=args.instance, name=args.username, host=args.host))
    password_policy = users.CreatePasswordPolicyFromArgs(sql_messages, user.passwordPolicy, args)
    result_operation = sql_client.users.Update(sql_messages.SqlUsersUpdateRequest(project=instance_ref.project, instance=args.instance, name=args.username, host=args.host, user=sql_messages.User(project=instance_ref.project, instance=args.instance, name=args.username, host=args.host, passwordPolicy=password_policy)))
    operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
    if args.async_:
        return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
    operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Updating Cloud SQL user')