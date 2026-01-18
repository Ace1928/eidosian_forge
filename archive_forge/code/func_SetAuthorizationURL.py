from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.certificate_manager import api_client
from googlecloudsdk.core.util import times
def SetAuthorizationURL(ref, args, request):
    """Converts the dns-authorization argument into a relative URL with project name and location.

  Args:
    ref: Reference to the membership object.
    args: Command line arguments.
    request: API request to be issued

  Returns:
    Modified request
  """
    del ref
    if not args:
        return request
    if args.dns_authorizations:
        authorizations = []
        for field in args.dns_authorizations:
            if not field.startswith('projects/'):
                authorizations.append(DNS_AUTHORIZATIONS_TEMPLATE.format(request.parent, field))
            else:
                authorizations.append(field)
        request.certificate.managed.dnsAuthorizations = authorizations
    return request