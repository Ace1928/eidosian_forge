from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def UpdateRoles(unused_ref, args, request):
    """Update 'MembershipRoles' to request.modifyMembershipRolesRequest.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.

  """
    if hasattr(args, 'add_roles') and args.IsSpecified('add_roles'):
        role_list = args.add_roles.split(',')
        version = groups_hooks.GetApiVersion(args)
        roles = []
        messages = ci_client.GetMessages(version)
        for role in role_list:
            membership_role = messages.MembershipRole(name=role)
            roles.append(membership_role)
        request.modifyMembershipRolesRequest = messages.ModifyMembershipRolesRequest(addRoles=roles)
    return request