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
def UpdateDescription(unused_ref, args, request):
    """Update description.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    if args.IsSpecified('clear_description'):
        request.group.description = ''
    elif args.IsSpecified('description'):
        request.group.description = args.description
    return request