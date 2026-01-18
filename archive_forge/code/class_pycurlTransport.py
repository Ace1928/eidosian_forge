import logging
import ssl
import sys
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, Struct
class pycurlTransport(TransportBase):
    _wrapper_version = pycurl.version
    _wrapper_name = 'pycurl'

    def __init__(self, timeout, proxy=None, cacert=None, sessions=False):
        self.timeout = timeout
        self.proxy = proxy or {}
        self.cacert = cacert

    def request(self, url, method, body, headers):
        c = pycurl.Curl()
        c.setopt(pycurl.URL, url)
        if 'proxy_host' in self.proxy:
            c.setopt(pycurl.PROXY, self.proxy['proxy_host'])
        if 'proxy_port' in self.proxy:
            c.setopt(pycurl.PROXYPORT, self.proxy['proxy_port'])
        if 'proxy_user' in self.proxy:
            c.setopt(pycurl.PROXYUSERPWD, '%(proxy_user)s:%(proxy_pass)s' % self.proxy)
        self.buf = StringIO()
        c.setopt(pycurl.WRITEFUNCTION, self.buf.write)
        if self.cacert:
            c.setopt(c.CAINFO, self.cacert)
        c.setopt(pycurl.SSL_VERIFYPEER, self.cacert and 1 or 0)
        c.setopt(pycurl.SSL_VERIFYHOST, self.cacert and 2 or 0)
        c.setopt(pycurl.CONNECTTIMEOUT, self.timeout)
        c.setopt(pycurl.TIMEOUT, self.timeout)
        if method == 'POST':
            c.setopt(pycurl.POST, 1)
            c.setopt(pycurl.POSTFIELDS, body)
        if headers:
            hdrs = ['%s: %s' % (k, v) for k, v in headers.items()]
            log.debug(hdrs)
            c.setopt(pycurl.HTTPHEADER, hdrs)
        c.perform()
        c.close()
        return ({}, self.buf.getvalue())