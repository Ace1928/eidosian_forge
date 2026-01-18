from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
def ParseDualPasswordType(sql_messages, args):
    """Parses the correct retained password type for the arguments given.

  Args:
    sql_messages: the proto definition for the API being called
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    DualPasswordType enum or None
  """
    if args.discard_dual_password:
        return sql_messages.User.DualPasswordTypeValueValuesEnum.NO_DUAL_PASSWORD
    if args.retain_password:
        return sql_messages.User.DualPasswordTypeValueValuesEnum.DUAL_PASSWORD
    return None