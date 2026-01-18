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
def ParseResource(request):
    """Returns an updated `request` with the resource path parsed."""
    resource, resource_name = GetResourceInfo(request)
    new_resource = resource[resource.rindex(PROJECTS_RESOURCE_PATH):]
    setattr(request, resource_name, new_resource)
    return request