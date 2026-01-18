from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def SetMembershipResourceName(unused_ref, args, request):
    """Set membership resource name to request.name.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.

  """
    version = groups_hooks.GetApiVersion(args)
    name = ''
    if args.IsSpecified('group_email') and args.IsSpecified('member_email'):
        name = ConvertEmailToMembershipResourceName(version, args, '--group-email', '--member-email')
    else:
        raise exceptions.InvalidArgumentException('Must specify `--group-email` and `--member-email` argument.')
    request.name = name
    if hasattr(request, 'membership'):
        request.membership.name = name
    return request