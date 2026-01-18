from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
from six.moves import urllib
def GetDefaultProxyInfo(method='http'):
    """Get ProxyInfo from environment.

  This function is meant to mimic httplib2.proxy_info_from_environment, but get
  the proxy information from urllib.getproxies instead. urllib can also get
  proxy information from Windows Internet Explorer settings or MacOSX framework
  SystemConfiguration.

  Args:
    method: protocol string
  Returns:
    httplib2 ProxyInfo object or None
  """
    proxy_dict = urllib.request.getproxies()
    proxy_url = proxy_dict.get(method, None)
    if not proxy_url:
        return None
    pi = httplib2.proxy_info_from_url(proxy_url, method)
    pi.bypass_host = urllib.request.proxy_bypass
    return pi