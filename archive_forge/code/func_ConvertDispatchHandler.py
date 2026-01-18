from __future__ import absolute_import
import re
def ConvertDispatchHandler(handler):
    """Create conversion function which handles dispatch rules.

  Extract domain and path from dispatch url,
  set service value from service or module info.

  Args:
    handler: Result of converting handler according to schema.

  Returns:
    Handler which has 'domain', 'path' and 'service' fields.
  """
    dispatch_url = dispatchinfo.ParsedURL(handler['url'])
    dispatch_service = handler['service']
    dispatch_domain = dispatch_url.host
    if not dispatch_url.host_exact:
        dispatch_domain = '*' + dispatch_domain
    dispatch_path = dispatch_url.path
    if not dispatch_url.path_exact:
        dispatch_path = dispatch_path.rstrip('/') + '/*'
    new_handler = {}
    new_handler['domain'] = dispatch_domain
    new_handler['path'] = dispatch_path
    new_handler['service'] = dispatch_service
    return new_handler