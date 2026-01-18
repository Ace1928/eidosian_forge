import logging
import ssl
import sys
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, Struct
class Httplib2Transport(httplib2.Http, TransportBase):
    _wrapper_version = 'httplib2 %s' % httplib2.__version__
    _wrapper_name = 'httplib2'

    def __init__(self, timeout, proxy=None, cacert=None, sessions=False):
        kwargs = {}
        if proxy:
            import socks
            kwargs['proxy_info'] = httplib2.ProxyInfo(proxy_type=socks.PROXY_TYPE_HTTP, **proxy)
            log.info('using proxy %s' % proxy)
        if httplib2.__version__ >= '0.3.0':
            kwargs['timeout'] = timeout
        if httplib2.__version__ >= '0.7.0':
            kwargs['disable_ssl_certificate_validation'] = cacert is None
            kwargs['ca_certs'] = cacert
        httplib2.Http.__init__(self, **kwargs)