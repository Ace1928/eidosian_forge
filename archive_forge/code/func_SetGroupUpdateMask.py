from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def SetGroupUpdateMask(unused_ref, args, request):
    """Set the update mask on the request based on the args.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  Raises:
    InvalidArgumentException: If no fields are specified to update.
  """
    update_mask = []
    if args.IsSpecified('display_name') or args.IsSpecified('clear_display_name'):
        update_mask.append('display_name')
    if args.IsSpecified('description') or args.IsSpecified('clear_description'):
        update_mask.append('description')
    if hasattr(args, 'labels'):
        if args.IsSpecified('labels'):
            update_mask.append('labels')
    if hasattr(args, 'add_posix_group'):
        if args.IsSpecified('add_posix_group') or args.IsSpecified('remove_posix_groups') or args.IsSpecified('clear_posix_groups'):
            update_mask.append('posix_groups')
    if args.IsSpecified('dynamic_user_query'):
        update_mask.append('dynamic_group_metadata')
    if not update_mask:
        raise exceptions.InvalidArgumentException('Must specify at least one field mask.')
    request.updateMask = ','.join(update_mask)
    return request