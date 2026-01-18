from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
def RunBaseListCommand(args, release_track):
    """Lists Cloud SQL users in a given instance.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    release_track: base.ReleaseTrack, the release track that this was run under.

  Returns:
    SQL user resource iterator.
  """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    project_id = properties.VALUES.core.project.Get(required=True)
    users = sql_client.users.List(sql_messages.SqlUsersListRequest(project=project_id, instance=args.instance)).items
    dual_password_type = ''
    for user in users:
        if user.dualPasswordType:
            dual_password_type = 'dualPasswordType,'
        policy = user.passwordPolicy
        if not policy:
            continue
        policy.enableFailedAttemptsCheck = None
    if release_track == base.ReleaseTrack.GA:
        args.GetDisplayInfo().AddFormat("\n      table(\n        name.yesno(no='(anonymous)'),\n        host,\n        type.yesno(no='BUILT_IN'),\n        {dualPasswordType}\n        passwordPolicy\n      )\n    ".format(dualPasswordType=dual_password_type))
    else:
        args.GetDisplayInfo().AddFormat("\n      table(\n        name.yesno(no='(anonymous)'),\n        host,\n        type.yesno(no='BUILT_IN'),\n        {dualPasswordType}\n        iamEmail,\n        passwordPolicy\n      )\n    ".format(dualPasswordType=dual_password_type))
    return users