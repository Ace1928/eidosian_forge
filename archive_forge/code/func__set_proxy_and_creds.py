import base64
from urllib.parse import unquote, urlunparse
from urllib.request import _parse_proxy, getproxies, proxy_bypass
from scrapy.exceptions import NotConfigured
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes
def _set_proxy_and_creds(self, request, proxy_url, creds):
    if proxy_url:
        request.meta['proxy'] = proxy_url
    elif request.meta.get('proxy') is not None:
        request.meta['proxy'] = None
    if creds:
        request.headers[b'Proxy-Authorization'] = b'Basic ' + creds
        request.meta['_auth_proxy'] = proxy_url
    elif '_auth_proxy' in request.meta:
        if proxy_url != request.meta['_auth_proxy']:
            if b'Proxy-Authorization' in request.headers:
                del request.headers[b'Proxy-Authorization']
            del request.meta['_auth_proxy']
    elif b'Proxy-Authorization' in request.headers:
        if proxy_url:
            request.meta['_auth_proxy'] = proxy_url
        else:
            del request.headers[b'Proxy-Authorization']