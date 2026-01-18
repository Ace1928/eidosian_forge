from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from urllib import parse
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import http_proxy_types
import socks
def GetProxyHostPort(url):
    proxy_host = SECURITY_GATEWAY_PROXY_HOST
    proxy_port = SECURITY_GATEWAY_PROXY_PORT
    if url:
        info = parse.urlparse(url)
        proxy_host = info.hostname
        proxy_port = info.port
        if not proxy_host or not proxy_port:
            raise ValueError('{} is an invalid url'.format(url))
    return (proxy_host, proxy_port)