from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def AddExpiryDetailInMembershipRoles(version, request, expiration):
    """Add an expiration in request.membership.roles.

  Args:
    version: version
    request: The request to modify
    expiration: expiration date to set

  Returns:
    The updated roles.

  Raises:
    InvalidArgumentException: If 'expiration' is specified without MEMBER role.

  """
    messages = ci_client.GetMessages(version)
    roles = []
    has_member_role = False
    for role in request.membership.roles:
        if hasattr(role, 'name') and role.name == 'MEMBER':
            has_member_role = True
            roles.append(messages.MembershipRole(name='MEMBER', expiryDetail=ReformatExpiryDetail(version, expiration, 'add')))
        else:
            roles.append(role)
    if not has_member_role:
        raise exceptions.InvalidArgumentException('expiration', 'Expiration date can be set with a MEMBER role only.')
    return roles