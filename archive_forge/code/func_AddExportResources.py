from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def AddExportResources(resource_ref, args, request):
    """Sets export resource paths for an UpdateSubscriptionRequest.

  Args:
    resource_ref: resources.Resource, the resource reference for the resource
      being operated on.
    args: argparse.Namespace, the parsed commandline arguments.
    request: An UpdateSubscriptionRequest.

  Returns:
    The UpdateSubscriptionRequest.
  """
    del resource_ref
    if request.subscription.exportConfig is None:
        return request
    resource, _ = GetResourceInfo(request)
    project = DeriveProjectFromResource(resource)
    location = DeriveLocationFromResource(resource)
    psl = PubsubLiteMessages()
    SetExportConfigResources(args, psl, project, location, request.subscription.exportConfig)
    return request