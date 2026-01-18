from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
from six.moves import urllib
def GetHttpProxyInfo():
    """Get ProxyInfo object or callable to be passed to httplib2.Http.

  httplib2.Http can issue requests through a proxy. That information is passed
  via either ProxyInfo objects or a callback function that receives the protocol
  the request is made on and returns the proxy address. If users set the gcloud
  properties, we create a ProxyInfo object with those settings. If users do not
  set gcloud properties, we return a function that can be called to get default
  settings.

  Returns:
    httplib2 ProxyInfo object or callable function that returns a Proxy Info
    object given the protocol (http, https)
  """
    proxy_settings = GetProxyProperties()
    if proxy_settings:
        return httplib2.ProxyInfo(proxy_settings['proxy_type'], proxy_settings['proxy_address'], proxy_settings['proxy_port'], proxy_rdns=proxy_settings['proxy_rdns'], proxy_user=proxy_settings['proxy_user'], proxy_pass=proxy_settings['proxy_pass'])
    return GetDefaultProxyInfo