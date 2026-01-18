from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
def _NormalRequest(self, service, request):
    """Generates a basic request function for the method.

    Args:
      service: The apitools service that will be making the request.
      request: The apitools request object to send.

    Returns:
      A function to make the request.
    """

    def RequestFunc(global_params=None):
        method = getattr(service, self._method_name)
        return method(request, global_params=global_params)
    return RequestFunc