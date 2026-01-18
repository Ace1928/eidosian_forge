from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.command_lib.util.args import labels_util
def SetParentCollection(ref, args, request):
    """Set parent collection with location for created resources.

  Args:
    ref: reference to the scope object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del ref, args
    request.parent = request.parent + '/locations/-'
    return request