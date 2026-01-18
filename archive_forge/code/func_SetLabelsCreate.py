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
def SetLabelsCreate(unused_ref, args, request):
    """Set Labels to request.group.labels for the create command.

  Labels will be used from args.labels if supplied, otherwise labels
  will be looked up based on the args.group_type argument. If neither is
  supplied, labels will be set based on the 'discussion' group type.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    if args.IsSpecified('labels'):
        labels = args.labels
    elif args.IsKnownAndSpecified('group_type'):
        labels = ','.join(GROUP_TYPE_MAP[args.group_type])
    else:
        labels = ','.join(GROUP_TYPE_MAP['discussion'])
    if hasattr(request.group, 'labels'):
        request.group.labels = ReformatLabels(args, labels)
    else:
        version = GetApiVersion(args)
        messages = ci_client.GetMessages(version)
        request.group = messages.Group(labels=ReformatLabels(args, labels))
    return request