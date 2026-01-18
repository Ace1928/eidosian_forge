from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def CreateSetPasswordRequest(sql_messages, args, dual_password_type, project):
    """Creates the set password request to send to UpdateUser.

  Args:
    sql_messages: module, The messages module that should be used.
    args: argparse.Namespace, The arguments that this command was invoked with.
    dual_password_type: How we want to interact with the dual password.
    project: the project that this user is in

  Returns:
    CreateSetPasswordRequest
  """
    user = sql_messages.User(project=project, instance=args.instance, name=args.username, host=args.host, password=args.password)
    if dual_password_type:
        user.dualPasswordType = dual_password_type
    return sql_messages.SqlUsersUpdateRequest(project=project, instance=args.instance, name=args.username, host=args.host, user=user)