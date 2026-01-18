from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def SetMembershipParent(unused_ref, args, request):
    """Set resource name to request.parent.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.

  """
    version = groups_hooks.GetApiVersion(args)
    if args.IsSpecified('group_email'):
        request.parent = groups_hooks.ConvertEmailToResourceName(version, args.group_email, '--group-email')
    return request