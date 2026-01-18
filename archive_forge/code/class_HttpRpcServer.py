import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
class HttpRpcServer(AbstractRpcServer):
    """Provides a simplified RPC-style interface for HTTP requests."""
    DEFAULT_COOKIE_FILE_PATH = '~/.appcfg_cookies'

    def __init__(self, *args, **kwargs):
        self.certpath = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lib', 'cacerts', 'cacerts.txt'))
        self.cert_file_available = not kwargs.get('ignore_certs', False) and os.path.exists(self.certpath)
        super(HttpRpcServer, self).__init__(*args, **kwargs)

    def _CreateRequest(self, url, data=None):
        """Creates a new urllib request."""
        req = super(HttpRpcServer, self)._CreateRequest(url, data)
        if self.cert_file_available and can_validate_certs():
            req.set_ssl_info(ca_certs=self.certpath)
        return req

    def _CheckCookie(self):
        """Warn if cookie is not valid for at least one minute."""
        min_expire = time.time() + 60
        for cookie in self.cookie_jar:
            if cookie.domain == self.host and (not cookie.is_expired(min_expire)):
                break
        else:
            (print >> sys.stderr, '\nError: Machine system clock is incorrect.\n')

    def _Authenticate(self):
        """Save the cookie jar after authentication."""
        if self.cert_file_available and (not can_validate_certs()):
            logger.warn('ssl module not found.\nWithout the ssl module, the identity of the remote host cannot be verified, and\nconnections may NOT be secure. To fix this, please install the ssl module from\nhttp://pypi.python.org/pypi/ssl .\nTo learn more, see https://developers.google.com/appengine/kb/general#rpcssl')
        super(HttpRpcServer, self)._Authenticate()
        if self.cookie_jar.filename is not None and self.save_cookies:
            logger.debug('Saving authentication cookies to %s', self.cookie_jar.filename)
            self.cookie_jar.save()
            self._CheckCookie()

    def _GetOpener(self):
        """Returns an OpenerDirector that supports cookies and ignores redirects.

    Returns:
      A urllib2.OpenerDirector object.
    """
        opener = OpenerDirector()
        opener.add_handler(ProxyHandler())
        opener.add_handler(UnknownHandler())
        opener.add_handler(HTTPHandler())
        opener.add_handler(HTTPDefaultErrorHandler())
        opener.add_handler(HTTPSHandler())
        opener.add_handler(HTTPErrorProcessor())
        opener.add_handler(ContentEncodingHandler())
        if self.save_cookies:
            self.cookie_jar.filename = os.path.expanduser(HttpRpcServer.DEFAULT_COOKIE_FILE_PATH)
            if os.path.exists(self.cookie_jar.filename):
                try:
                    self.cookie_jar.load()
                    self.authenticated = True
                    logger.debug('Loaded authentication cookies from %s', self.cookie_jar.filename)
                except (OSError, IOError, LoadError) as e:
                    logger.debug('Could not load authentication cookies; %s: %s', e.__class__.__name__, e)
                    self.cookie_jar.filename = None
            else:
                try:
                    fd = os.open(self.cookie_jar.filename, os.O_CREAT, 384)
                    os.close(fd)
                except (OSError, IOError) as e:
                    logger.debug('Could not create authentication cookies file; %s: %s', e.__class__.__name__, e)
                    self.cookie_jar.filename = None
        opener.add_handler(HTTPCookieProcessor(self.cookie_jar))
        return opener