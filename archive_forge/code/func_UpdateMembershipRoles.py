from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def UpdateMembershipRoles(unused_ref, args, request):
    """Update MembershipRoles to request.membership.roles.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.

  """
    version = groups_hooks.GetApiVersion(args)
    if hasattr(args, 'roles') and args.IsSpecified('roles'):
        request.membership.roles = ReformatMembershipRoles(version, args.roles)
    return request