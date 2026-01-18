from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def TokenizeUpdateRolesParams(update_roles_param, arg_name):
    """Tokenize update_roles_params string.

  Args:
    update_roles_param: 'update_roles_param' string (e.g. MEMBER=expiration=3d)
    arg_name: The argument name

  Returns:
    Tokenized strings: role (e.g. MEMBER), param_key (e.g. expiration), and
    param_value (e.g. 3d)

  Raises:
    InvalidArgumentException: If invalid update_roles_param string is input.
  """
    token_list = update_roles_param.split('=')
    if len(token_list) == 3:
        return (token_list[0], token_list[1], token_list[2])
    raise exceptions.InvalidArgumentException(arg_name, 'Invalid format: ' + update_roles_param)