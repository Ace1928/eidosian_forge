from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def ReformatMembershipRoles(version, roles_list):
    """Reformat roles string to MembershipRoles object list.

  Args:
    version: Release track information
    roles_list: list of roles in a string format.

  Returns:
    List of MembershipRoles object.

  """
    messages = ci_client.GetMessages(version)
    roles = []
    if not roles_list:
        roles.append(messages.MembershipRole(name='MEMBER'))
        return roles
    for role in roles_list:
        new_membership_role = messages.MembershipRole(name=role)
        roles.append(new_membership_role)
    return roles