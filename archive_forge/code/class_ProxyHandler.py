import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class ProxyHandler(urllib.request.ProxyHandler):
    """Handles proxy setting.

    Copied and modified from urllib.request to be able to modify the request during
    the request pre-processing instead of modifying it at _open time. As we
    capture (or create) the connection object during request processing, _open
    time was too late.

    The main task is to modify the request so that the connection is done to
    the proxy while the request still refers to the destination host.

    Note: the proxy handling *may* modify the protocol used; the request may be
    against an https server proxied through an http proxy. So, https_request
    will be called, but later it's really http_open that will be called. This
    explains why we don't have to call self.parent.open as the urllib.request did.
    """
    handler_order = 100
    _debuglevel = DEBUG

    def __init__(self, proxies=None):
        urllib.request.ProxyHandler.__init__(self, proxies)
        for type, proxy in self.proxies.items():
            if self._debuglevel >= 3:
                print('Will unbind {}_open for {!r}'.format(type, proxy))
            delattr(self, '%s_open' % type)

        def bind_scheme_request(proxy, scheme):
            if proxy is None:
                return
            scheme_request = scheme + '_request'
            if self._debuglevel >= 3:
                print('Will bind {} for {!r}'.format(scheme_request, proxy))
            setattr(self, scheme_request, lambda request: self.set_proxy(request, scheme))
        http_proxy = self.get_proxy_env_var('http')
        bind_scheme_request(http_proxy, 'http')
        https_proxy = self.get_proxy_env_var('https')
        bind_scheme_request(https_proxy, 'https')

    def get_proxy_env_var(self, name, default_to='all'):
        """Get a proxy env var.

        Note that we indirectly rely on
        urllib.getproxies_environment taking into account the
        uppercased values for proxy variables.
        """
        try:
            return self.proxies[name.lower()]
        except KeyError:
            if default_to is not None:
                try:
                    return self.proxies[default_to]
                except KeyError:
                    pass
        return None

    def proxy_bypass(self, host):
        """Check if host should be proxied or not.

        :returns: True to skip the proxy, False otherwise.
        """
        no_proxy = self.get_proxy_env_var('no', default_to=None)
        bypass = self.evaluate_proxy_bypass(host, no_proxy)
        if bypass is None:
            return urllib.request.proxy_bypass(host)
        else:
            return bypass

    def evaluate_proxy_bypass(self, host, no_proxy):
        """Check the host against a comma-separated no_proxy list as a string.

        :param host: ``host:port`` being requested

        :param no_proxy: comma-separated list of hosts to access directly.

        :returns: True to skip the proxy, False not to, or None to
            leave it to urllib.
        """
        if no_proxy is None:
            return False
        hhost, hport = splitport(host)
        for domain in no_proxy.split(','):
            domain = domain.strip()
            if domain == '':
                continue
            dhost, dport = splitport(domain)
            if hport == dport or dport is None:
                dhost = dhost.replace('.', '\\.')
                dhost = dhost.replace('*', '.*')
                dhost = dhost.replace('?', '.')
                if re.match(dhost, hhost, re.IGNORECASE):
                    return True
        return None

    def set_proxy(self, request, type):
        host = request.host
        if self.proxy_bypass(host):
            return request
        proxy = self.get_proxy_env_var(type)
        if self._debuglevel >= 3:
            print('set_proxy {}_request for {!r}'.format(type, proxy))
        parsed_url = transport.ConnectedTransport._split_url(proxy)
        if not parsed_url.host:
            raise urlutils.InvalidURL(proxy, 'No host component')
        if request.proxy_auth == {}:
            request.proxy_auth = {'host': parsed_url.host, 'port': parsed_url.port, 'user': parsed_url.user, 'password': parsed_url.password, 'protocol': parsed_url.scheme, 'path': None}
        if parsed_url.port is None:
            phost = parsed_url.host
        else:
            phost = parsed_url.host + ':%d' % parsed_url.port
        request.set_proxy(phost, type)
        if self._debuglevel >= 3:
            print('set_proxy: proxy set to {}://{}'.format(type, phost))
        return request