import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
def _parse_proxy_url(self, proxy_url):
    """
        Parse and validate a proxy URL.

        :param proxy_url: Proxy URL (e.g. http://hostname:3128)
        :type proxy_url: ``str``

        :rtype: ``tuple`` (``scheme``, ``hostname``, ``port``)
        """
    parsed = urlparse.urlparse(proxy_url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError('Only http and https proxies are supported')
    if not parsed.hostname or not parsed.port:
        raise ValueError('proxy_url must be in the following format: <scheme>://<proxy host>:<proxy port>')
    proxy_scheme = parsed.scheme
    proxy_host, proxy_port = (parsed.hostname, parsed.port)
    netloc = parsed.netloc
    if '@' in netloc:
        username_password = netloc.split('@', 1)[0]
        split = username_password.split(':', 1)
        if len(split) < 2:
            raise ValueError('URL is in an invalid format')
        proxy_username, proxy_password = (split[0], split[1])
    else:
        proxy_username = None
        proxy_password = None
    return (proxy_scheme, proxy_host, proxy_port, proxy_username, proxy_password)