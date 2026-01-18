from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.certificate_manager import api_client
from googlecloudsdk.core.util import times
def SetCAPoolURL(ref, args, request):
    """Converts the ca-pool argument into a relative URL with project name and location.

  Args:
    ref: reference to the membership object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del ref
    if not args:
        return request
    if args.ca_pool:
        if not args.ca_pool.startswith('projects/'):
            request.certificateIssuanceConfig.certificateAuthorityConfig.certificateAuthorityServiceConfig.caPool = CA_POOL_TEMPLATE.format(request.parent, args.ca_pool)
    return request