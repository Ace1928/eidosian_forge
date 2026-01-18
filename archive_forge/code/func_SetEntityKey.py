from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def SetEntityKey(unused_ref, args, request):
    """Set EntityKey in group resource.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.

  """
    version = groups_hooks.GetApiVersion(args)
    messages = ci_client.GetMessages(version)
    if hasattr(args, 'member_email') and args.IsSpecified('member_email'):
        entity_key = messages.EntityKey(id=args.member_email)
        request.membership.preferredMemberKey = entity_key
    return request