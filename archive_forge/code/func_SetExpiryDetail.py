from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def SetExpiryDetail(unused_ref, args, request):
    """Set expiration to request.membership.expiryDetail (v1alpha1) or in request.membership.roles (v1beta1).

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.

  Raises:
    InvalidArgumentException: If 'expiration' is specified upon following cases:
    1. 'request.membership' doesn't have 'roles' attribute, or
    2. multiple roles are provided.

  """
    if not hasattr(request.membership, 'roles'):
        raise exceptions.InvalidArgumentException('expiration', 'roles must be specified.')
    if len(request.membership.roles) != 1:
        raise exceptions.InvalidArgumentException('roles', 'When setting "expiration", a single role should be input.')
    version = groups_hooks.GetApiVersion(args)
    if hasattr(args, 'expiration') and args.IsSpecified('expiration'):
        if version == 'v1alpha1':
            request.membership.expiryDetail = ReformatExpiryDetail(version, args.expiration, 'add')
        else:
            request.membership.roles = AddExpiryDetailInMembershipRoles(version, request, args.expiration)
    return request