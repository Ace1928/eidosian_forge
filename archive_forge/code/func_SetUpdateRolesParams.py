from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def SetUpdateRolesParams(unused_ref, args, request):
    """Update 'MembershipRoles' to request.modifyMembershipRolesRequest.

  Args:
    unused_ref: A string representing the operation reference. Unused and may
      be None.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.

  """
    if hasattr(args, 'update_roles_params') and args.IsSpecified('update_roles_params'):
        version = groups_hooks.GetApiVersion(args)
        messages = ci_client.GetMessages(version)
        request.modifyMembershipRolesRequest = messages.ModifyMembershipRolesRequest(updateRolesParams=ReformatUpdateRolesParams(args, args.update_roles_params))
    return request