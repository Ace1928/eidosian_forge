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
def SetParent(unused_ref, args, request):
    """Set obfuscated customer id to request.group.parent or request.parent.

  Args:
    unused_ref: A string representing the operation reference. Unused and may be
      None.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    version = GetApiVersion(args)
    messages = ci_client.GetMessages(version)
    group = getattr(request, 'group', None)
    if group is None:
        request.group = messages.Group()
    request.group.parent = GetCustomerId(args)
    return request