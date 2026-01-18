import logging
import ssl
import sys
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, Struct
class urllib2Transport(TransportBase):
    _wrapper_version = 'urllib2 %s' % urllib2.__version__
    _wrapper_name = 'urllib2'

    def __init__(self, timeout=None, proxy=None, cacert=None, sessions=False):
        if timeout is not None and (not self.supports_feature('timeout')):
            raise RuntimeError('timeout is not supported with urllib2 transport')
        if proxy:
            raise RuntimeError('proxy is not supported with urllib2 transport')
        if cacert:
            raise RuntimeError('cacert is not support with urllib2 transport')
        handlers = []
        if sys.version_info[0] == 2 and sys.version_info >= (2, 7, 9) or (sys.version_info[0] == 3 and sys.version_info >= (3, 2, 0)):
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib2.HTTPSHandler(context=context))
        if sessions:
            handlers.append(urllib2.HTTPCookieProcessor(CookieJar()))
        opener = urllib2.build_opener(*handlers)
        self.request_opener = opener.open
        self._timeout = timeout

    def request(self, url, method='GET', body=None, headers={}):
        req = urllib2.Request(url, body, headers)
        try:
            f = self.request_opener(req, timeout=self._timeout)
            return (f.info(), f.read())
        except urllib2.HTTPError as f:
            if f.code != 500:
                raise
            return (f.info(), f.read())